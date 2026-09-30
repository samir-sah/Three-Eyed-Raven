"""CLI for evaluating a tracker against labeled JSONL annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .tracking_evaluation import evaluate_tracking, parse_labeled_box


def _load_jsonl(path: str, source_name: str):
    records = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in {source_name} line {line_number}.") from error
        records.append(parse_labeled_box(raw, f"{source_name} line {line_number}"))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate JSONL tracker output against labeled ground truth.")
    parser.add_argument("--ground-truth", required=True, help="JSONL ground-truth annotations with stable IDs.")
    parser.add_argument("--predictions", required=True, help="JSONL tracker observations to evaluate.")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for frame matching.")
    parser.add_argument("--output", help="Optional path for the JSON metric report.")
    args = parser.parse_args()

    metrics = evaluate_tracking(
        _load_jsonl(args.ground_truth, "ground truth"),
        _load_jsonl(args.predictions, "predictions"),
        args.iou_threshold,
    )
    rendered = json.dumps(metrics, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
