"""Convert MOTChallenge ground truth to this project's labeled JSONL format."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MOTChallenge gt.txt annotations to labeled JSONL.")
    parser.add_argument("--input", required=True, help="MOTChallenge gt.txt CSV path.")
    parser.add_argument("--output", required=True, help="Destination JSONL path.")
    parser.add_argument("--person-class", type=int, default=1, help="MOT class ID to retain (default: 1).")
    parser.add_argument("--min-visibility", type=float, default=0.0, help="Minimum MOT visibility to retain.")
    args = parser.parse_args()
    if not 0 <= args.min_visibility <= 1:
        raise ValueError("min-visibility must be between 0 and 1.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with Path(args.input).open("r", encoding="utf-8", newline="") as source, output.open("w", encoding="utf-8") as destination:
        for row_number, row in enumerate(csv.reader(source), start=1):
            if len(row) < 6:
                raise ValueError(f"MOT row {row_number} has fewer than six columns.")
            frame, identity = int(row[0]), int(row[1])
            left, top, width, height = (float(value) for value in row[2:6])
            confidence = float(row[6]) if len(row) > 6 else 1.0
            category = int(float(row[7])) if len(row) > 7 else args.person_class
            visibility = float(row[8]) if len(row) > 8 else 1.0
            if confidence <= 0 or category != args.person_class or visibility < args.min_visibility:
                continue
            destination.write(json.dumps({
                "frame_index": frame - 1,
                "track_id": str(identity),
                "bbox_xyxy": [left, top, left + width, top + height],
            }) + "\n")
            written += 1
    print(json.dumps({"written": written, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
