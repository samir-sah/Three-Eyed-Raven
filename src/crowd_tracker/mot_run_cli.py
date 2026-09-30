"""Run the project's YOLO + ByteTrack pipeline directly on a MOT image sequence."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from .models import TrackObservation
from .profiling import StageProfiler
from .storage import RunStore
from .tracker import ByteTrackPersonTracker


def main() -> None:
    parser = argparse.ArgumentParser(description="Track people in a MOTChallenge img1 sequence.")
    parser.add_argument("--images", required=True, help="MOT sequence img1 directory.")
    parser.add_argument("--output", required=True, help="Artifact directory for observations and summary.")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    image_paths = sorted(path for path in Path(args.images).iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.max_frames is not None:
        image_paths = image_paths[:args.max_frames]
    if not image_paths:
        raise ValueError("No MOT sequence images found.")
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    tracker = ByteTrackPersonTracker(args.model, args.device, args.confidence, args.iou, args.tracker)
    profiler = StageProfiler()
    unique_ids: set[int] = set()
    started = time.perf_counter()
    with (destination / "observations.jsonl").open("w", encoding="utf-8") as events:
        for index, image_path in enumerate(image_paths):
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            with profiler.measure("detection_and_tracking"):
                tracked = tracker.track(frame)
            for item in tracked:
                unique_ids.add(item.track_id)
                events.write(json.dumps(TrackObservation(
                    camera_id="mot_sequence",
                    frame_index=index,
                    timestamp_seconds=float(index),
                    track_id=item.track_id,
                    bbox_xyxy=item.bbox_xyxy,
                    confidence=item.confidence,
                    zones=(),
                ).as_dict()) + "\n")
    elapsed = max(time.perf_counter() - started, 1e-9)
    summary = {
        "run_type": "mot_benchmark",
        "source": str(args.images),
        "frames_processed": len(image_paths),
        "elapsed_seconds": round(elapsed, 3),
        "processing_fps": round(len(image_paths) / elapsed, 3),
        "unique_local_track_ids": len(unique_ids),
        "alert_count": 0,
        "latency_profile": profiler.summary(),
        "config": vars(args),
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    RunStore.for_output_dir(destination).upsert(destination.name, destination, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
