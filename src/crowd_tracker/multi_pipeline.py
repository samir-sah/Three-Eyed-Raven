"""Interleaved offline processing for independent aerial and CCTV streams."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from .analytics import ZoneAnalytics
from .config import CameraStreamConfig, MultiStreamConfig
from .models import TrackObservation
from .reid import AppearanceGallery, CrossCameraMatcher
from .tracker import ByteTrackPersonTracker, TrackedPerson
from .storage import RunStore


@dataclass
class _CameraState:
    config: CameraStreamConfig
    capture: cv2.VideoCapture
    tracker: ByteTrackPersonTracker
    analytics: ZoneAnalytics
    output_dir: Path
    observations_file: object
    alerts_file: object
    source_fps: float
    source_width: int
    source_height: int
    writer: cv2.VideoWriter | None = None
    read: int = 0
    processed: int = 0
    finished: bool = False
    reconnects: int = 0
    alert_count: int = 0
    unique_track_ids: set[int] = field(default_factory=set)


class MultiStreamPipeline:
    """Processes one frame from each source per cycle without mixing local IDs."""

    def __init__(self, config: MultiStreamConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gallery = AppearanceGallery(
            min_box_size=config.reid_min_box_size,
            max_samples_per_track=6,
        )
        self.matcher = CrossCameraMatcher(
            candidate_threshold=config.reid_candidate_threshold,
            top_k=config.reid_top_k,
        )

    def run(self) -> dict:
        states = [self._open_camera(camera) for camera in self.config.cameras]
        started = time.perf_counter()
        try:
            while any(not state.finished for state in states):
                for state in states:
                    if not state.finished:
                        self._process_next_frame(state)
        finally:
            for state in states:
                state.capture.release()
                if state.writer is not None:
                    state.writer.release()
                state.observations_file.close()
                state.alerts_file.close()

        elapsed = max(time.perf_counter() - started, 1e-9)
        cameras = [self._camera_summary(state, elapsed) for state in states]
        thumbnail_paths = self.gallery.write_thumbnails(str(self.output_dir / "reid_crops"))
        candidates = self.matcher.rank(self.gallery.descriptors(thumbnail_paths))
        reid_summary = {
            "method": "HSV appearance baseline",
            "candidate_threshold": self.config.reid_candidate_threshold,
            "review_candidates": sum(candidate["status"] == "review" for candidate in candidates),
            "ranked_pairs": len(candidates),
            "warning": "Appearance similarity is not identity verification. Review candidates manually; do not use this output for decisions about people.",
        }
        (self.output_dir / "reid_candidates.json").write_text(
            json.dumps({"summary": reid_summary, "candidates": candidates}, indent=2),
            encoding="utf-8",
        )
        summary = {
            "run_type": "multi_stream",
            "elapsed_seconds": round(elapsed, 3),
            "total_frames_processed": sum(camera["frames_processed"] for camera in cameras),
            "aggregate_processing_fps": round(sum(camera["frames_processed"] for camera in cameras) / elapsed, 3),
            "alert_count": sum(camera["alert_count"] for camera in cameras),
            "cameras": cameras,
            "reid": reid_summary,
            "config": self.config.as_dict(),
        }
        (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (self.output_dir / "manifest.json").write_text(json.dumps({"cameras": cameras}, indent=2), encoding="utf-8")
        RunStore.for_output_dir(self.output_dir).upsert(self.output_dir.name, self.output_dir, summary)
        return summary

    def _open_camera(self, camera: CameraStreamConfig) -> _CameraState:
        capture = cv2.VideoCapture(camera.source)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open source for {camera.id}: {camera.source}")
        output_dir = self.output_dir / camera.id
        output_dir.mkdir(parents=True, exist_ok=True)
        source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = None
        if self.config.save_video:
            writer = cv2.VideoWriter(
                str(output_dir / "annotated.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                source_fps,
                (source_width, source_height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not create output video for {camera.id}")
        return _CameraState(
            config=camera,
            capture=capture,
            tracker=ByteTrackPersonTracker(
                model_path=self.config.model,
                device=self.config.device,
                confidence=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                tracker=self.config.tracker,
            ),
            analytics=ZoneAnalytics(camera.zones),
            output_dir=output_dir,
            observations_file=(output_dir / "observations.jsonl").open("w", encoding="utf-8"),
            alerts_file=(output_dir / "alerts.jsonl").open("w", encoding="utf-8"),
            source_fps=source_fps,
            source_width=source_width,
            source_height=source_height,
            writer=writer,
        )

    def _process_next_frame(self, state: _CameraState) -> None:
        ok, frame = state.capture.read()
        if not ok:
            if _is_live_source(state.config.source) and state.reconnects < self.config.reconnect_attempts:
                state.reconnects += 1
                state.capture.release()
                time.sleep(self.config.reconnect_delay_seconds)
                state.capture = cv2.VideoCapture(state.config.source)
                if state.capture.isOpened():
                    return
            state.finished = True
            return
        state.read += 1
        if state.read % self.config.frame_stride != 0:
            return

        timestamp = state.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        tracked = state.tracker.track(frame)
        observations = self._observations(state, tracked, timestamp)
        state.unique_track_ids.update(item.track_id for item in observations)
        for observation in observations:
            state.observations_file.write(json.dumps(observation.as_dict()) + "\n")
            self.gallery.add(frame, observation)

        alerts = state.analytics.alerts(observations)
        for alert in alerts:
            state.alerts_file.write(json.dumps(alert.as_dict()) + "\n")
        state.alert_count += len(alerts)

        self._draw(state, frame, tracked, observations, alerts)
        if state.writer is not None:
            state.writer.write(frame)
        state.processed += 1
        if self.config.max_frames_per_camera is not None and state.processed >= self.config.max_frames_per_camera:
            state.finished = True

    def _observations(self, state: _CameraState, tracked: list[TrackedPerson], timestamp: float) -> list[TrackObservation]:
        return [
            TrackObservation(
                camera_id=state.config.id,
                frame_index=state.processed,
                timestamp_seconds=timestamp,
                track_id=item.track_id,
                bbox_xyxy=item.bbox_xyxy,
                confidence=item.confidence,
                zones=state.analytics.zones_for_bbox(item.bbox_xyxy),
            )
            for item in tracked
        ]

    @staticmethod
    def _draw(state: _CameraState, frame, tracked: list[TrackedPerson], observations: list[TrackObservation], alerts) -> None:
        zone_counts: dict[str, int] = defaultdict(int)
        for observation in observations:
            for zone_id in observation.zones:
                zone_counts[zone_id] += 1
        state.analytics.draw(frame, zone_counts)
        for item in tracked:
            x1, y1, x2, y2 = item.bbox_xyxy
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"local-{item.track_id} {item.confidence:.2f}", (x1, max(22, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{state.config.id}: {len(tracked)} persons", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
        if alerts:
            alert = alerts[0]
            cv2.putText(frame, f"ALERT: {alert.zone_id} count {alert.count} >= {alert.threshold}", (16, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    @staticmethod
    def _camera_summary(state: _CameraState, elapsed: float) -> dict:
        summary = {
            "camera_id": state.config.id,
            "source": state.config.source,
            "source_fps": state.source_fps,
            "source_resolution": [state.source_width, state.source_height],
            "frames_read": state.read,
            "frames_processed": state.processed,
            "unique_local_track_ids": len(state.unique_track_ids),
            "alert_count": state.alert_count,
            "reconnects": state.reconnects,
            "processing_fps": round(state.processed / elapsed, 3),
        }
        (state.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


def _is_live_source(source: str) -> bool:
    return source.lower().startswith(("rtsp://", "rtsps://", "http://", "https://"))
