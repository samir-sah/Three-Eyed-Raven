"""Video processing pipeline and artifact creation."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import cv2

from .analytics import ZoneAnalytics
from .config import AppConfig
from .models import TrackObservation
from .tracker import ByteTrackPersonTracker, TrackedPerson


def _open_jsonl(path: Path):
    return path.open("w", encoding="utf-8")


class OfflinePipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analytics = ZoneAnalytics(config.zones)
        self.tracker = ByteTrackPersonTracker(
            model_path=config.model,
            device=config.device,
            confidence=config.confidence_threshold,
            iou=config.iou_threshold,
            tracker=config.tracker,
        )

    def run(self) -> dict:
        capture = cv2.VideoCapture(self.config.source)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video source: {self.config.source}")

        source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = self._create_writer(source_fps, source_width, source_height)
        observations_path = self.output_dir / "observations.jsonl"
        alerts_path = self.output_dir / "alerts.jsonl"
        processed = 0
        read = 0
        unique_track_ids: set[int] = set()
        started = time.perf_counter()

        try:
            with _open_jsonl(observations_path) as observation_file, _open_jsonl(alerts_path) as alert_file:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    read += 1
                    if read % self.config.frame_stride != 0:
                        continue
                    timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    tracked = self.tracker.track(frame)
                    observations = self._observations(tracked, processed, timestamp)
                    unique_track_ids.update(item.track_id for item in observations)
                    for item in observations:
                        observation_file.write(json.dumps(item.as_dict()) + "\n")

                    alerts = self.analytics.alerts(observations)
                    for alert in alerts:
                        alert_file.write(json.dumps(alert.as_dict()) + "\n")

                    self._draw(frame, tracked, observations, alerts)
                    if writer is not None:
                        writer.write(frame)
                    if self.config.display:
                        cv2.imshow("Crowd tracker", frame)
                        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                            break

                    processed += 1
                    if self.config.max_frames is not None and processed >= self.config.max_frames:
                        break
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            if self.config.display:
                cv2.destroyAllWindows()

        elapsed = max(time.perf_counter() - started, 1e-9)
        summary = {
            "source": self.config.source,
            "source_fps": source_fps,
            "source_resolution": [source_width, source_height],
            "frames_read": read,
            "frames_processed": processed,
            "elapsed_seconds": round(elapsed, 3),
            "processing_fps": round(processed / elapsed, 3),
            "unique_local_track_ids": len(unique_track_ids),
            "config": self.config.as_dict(),
        }
        (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _create_writer(self, fps: float, width: int, height: int):
        if not self.config.save_video:
            return None
        path = self.output_dir / "annotated.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create output video: {path}")
        return writer

    def _observations(self, tracked: list[TrackedPerson], frame_index: int, timestamp: float) -> list[TrackObservation]:
        return [
            TrackObservation(
                camera_id="camera_1",
                frame_index=frame_index,
                timestamp_seconds=timestamp,
                track_id=item.track_id,
                bbox_xyxy=item.bbox_xyxy,
                confidence=item.confidence,
                zones=self.analytics.zones_for_bbox(item.bbox_xyxy),
            )
            for item in tracked
        ]

    def _draw(self, frame, tracked: list[TrackedPerson], observations: list[TrackObservation], alerts) -> None:
        zone_counts: dict[str, int] = defaultdict(int)
        for observation in observations:
            for zone_id in observation.zones:
                zone_counts[zone_id] += 1
        self.analytics.draw(frame, zone_counts)
        for item in tracked:
            x1, y1, x2, y2 = item.bbox_xyxy
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"local-{item.track_id} {item.confidence:.2f}",
                (x1, max(22, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(frame, f"Persons: {len(tracked)}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
        if alerts:
            alert = alerts[0]
            cv2.putText(
                frame,
                f"ALERT: {alert.zone_id} count {alert.count} >= {alert.threshold}",
                (16, 66),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
            )
