"""Convert MOTChallenge ground truth to this project's labeled JSONL format."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def import_mot_ground_truth(
    input_path: str | Path,
    output_path: str | Path,
    person_class: int = 1,
    min_visibility: float = 0.0,
    start_frame: int = 1,
    end_frame: int | None = None,
) -> int:
    """Convert one MOTChallenge GT file to evaluator JSONL and return records written."""
    if not 0 <= min_visibility <= 1:
        raise ValueError("min_visibility must be between 0 and 1.")
    if start_frame < 1 or (end_frame is not None and end_frame < start_frame):
        raise ValueError("Frame range is invalid.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with Path(input_path).open("r", encoding="utf-8", newline="") as source, output.open("w", encoding="utf-8") as destination:
        for row_number, row in enumerate(csv.reader(source), start=1):
            if len(row) < 6:
                raise ValueError(f"MOT row {row_number} has fewer than six columns.")
            frame, identity = int(row[0]), int(row[1])
            left, top, width, height = (float(value) for value in row[2:6])
            confidence = float(row[6]) if len(row) > 6 else 1.0
            category = int(float(row[7])) if len(row) > 7 else person_class
            visibility = float(row[8]) if len(row) > 8 else 1.0
            if frame < start_frame or (end_frame is not None and frame > end_frame):
                continue
            if confidence <= 0 or category != person_class or visibility < min_visibility:
                continue
            destination.write(json.dumps({
                "frame_index": frame - 1,
                "track_id": str(identity),
                "bbox_xyxy": [left, top, left + width, top + height],
            }) + "\n")
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MOTChallenge gt.txt annotations to labeled JSONL.")
    parser.add_argument("--input", required=True, help="MOTChallenge gt.txt CSV path.")
    parser.add_argument("--output", required=True, help="Destination JSONL path.")
    parser.add_argument("--person-class", type=int, default=1, help="MOT class ID to retain (default: 1).")
    parser.add_argument("--min-visibility", type=float, default=0.0, help="Minimum MOT visibility to retain.")
    parser.add_argument("--start-frame", type=int, default=1, help="First one-based MOT frame to include.")
    parser.add_argument("--end-frame", type=int, help="Last one-based MOT frame to include.")
    args = parser.parse_args()
    written = import_mot_ground_truth(
        args.input,
        args.output,
        args.person_class,
        args.min_visibility,
        args.start_frame,
        args.end_frame,
    )
    print(json.dumps({"written": written, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
