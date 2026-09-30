"""Batch evaluation of the project tracker on MOTChallenge sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .mot_import_cli import import_mot_ground_truth
from .mot_run_cli import run_mot_sequence
from .tracking_eval_cli import _load_jsonl
from .tracking_evaluation import evaluate_tracking


def aggregate_metrics(sequence_metrics: list[dict]) -> dict:
    """Aggregate counts exactly and average HOTA by ground-truth detections."""
    if not sequence_metrics:
        return {}
    totals = {
        key: sum(metric.get(key, 0) or 0 for metric in sequence_metrics)
        for key in (
            "ground_truth_boxes", "predicted_boxes", "matches", "false_positives", "false_negatives",
            "id_switches", "id_true_positives", "id_false_positives", "id_false_negatives",
        )
    }
    gt_total = totals["ground_truth_boxes"]
    predicted_total = totals["predicted_boxes"]
    matches = totals["matches"]
    id_true_positives = totals["id_true_positives"]
    precision = matches / predicted_total if predicted_total else 0.0
    recall = matches / gt_total if gt_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    id_precision = id_true_positives / (id_true_positives + totals["id_false_positives"]) if id_true_positives + totals["id_false_positives"] else 0.0
    id_recall = id_true_positives / (id_true_positives + totals["id_false_negatives"]) if id_true_positives + totals["id_false_negatives"] else 0.0
    idf1 = 2 * id_precision * id_recall / (id_precision + id_recall) if id_precision + id_recall else 0.0
    weighted_hota = sum((metric.get("hota") or 0) * metric.get("ground_truth_boxes", 0) for metric in sequence_metrics) / gt_total if gt_total else 0.0
    weighted_motp = sum((metric.get("motp") or 0) * metric.get("matches", 0) for metric in sequence_metrics) / matches if matches else None
    return {
        **totals,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mota": round(1 - (totals["false_positives"] + totals["false_negatives"] + totals["id_switches"]) / gt_total, 4) if gt_total else None,
        "motp": round(weighted_motp, 4) if weighted_motp is not None else None,
        "id_precision": round(id_precision, 4),
        "id_recall": round(id_recall, 4),
        "idf1": round(idf1, 4),
        "hota": round(weighted_hota, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and evaluate the tracker across MOTChallenge sequences.")
    parser.add_argument("--mot-root", required=True, help="Directory containing MOT17 sequence folders.")
    parser.add_argument("--output", required=True, help="Directory for sequence artifacts and benchmark_summary.json.")
    parser.add_argument("--model", required=True, help="YOLO checkpoint to evaluate.")
    parser.add_argument("--sequences", nargs="*", help="Sequence folder names. Defaults to all *-FRCNN folders with gt.txt.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--max-frames", type=int, help="Optional cap for a repeatable smoke benchmark.")
    parser.add_argument("--min-visibility", type=float, default=0.0)
    args = parser.parse_args()

    mot_root = Path(args.mot_root)
    if not mot_root.is_dir():
        raise FileNotFoundError(f"MOT root not found: {mot_root}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    names = args.sequences or sorted(
        path.name for path in mot_root.iterdir()
        if path.is_dir() and path.name.endswith("-FRCNN") and (path / "img1").is_dir() and (path / "gt" / "gt.txt").is_file()
    )
    if not names:
        raise ValueError("No MOT sequences with img1 and gt/gt.txt were found.")

    reports = []
    for name in names:
        sequence = mot_root / name
        image_directory = sequence / "img1"
        gt_path = sequence / "gt" / "gt.txt"
        if not image_directory.is_dir() or not gt_path.is_file():
            raise FileNotFoundError(f"Sequence {name} needs img1 and gt/gt.txt.")
        sequence_output = output / name
        summary = run_mot_sequence(
            image_directory, sequence_output, args.model, args.device, args.confidence, args.iou, args.tracker, args.max_frames,
        )
        ground_truth_path = sequence_output / "ground_truth.jsonl"
        import_mot_ground_truth(gt_path, ground_truth_path, min_visibility=args.min_visibility, end_frame=args.max_frames)
        metrics = evaluate_tracking(
            _load_jsonl(str(ground_truth_path), "ground truth"),
            _load_jsonl(str(sequence_output / "observations.jsonl"), "predictions"),
            args.iou,
        )
        (sequence_output / "evaluation.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        reports.append({"sequence": name, "run": summary, "evaluation": metrics})

    benchmark = {"model": args.model, "sequences": reports, "aggregate": aggregate_metrics([report["evaluation"] for report in reports])}
    (output / "benchmark_summary.json").write_text(json.dumps(benchmark, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(benchmark, indent=2))


if __name__ == "__main__":
    main()
