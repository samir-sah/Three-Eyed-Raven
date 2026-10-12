"""Aggregate image counts only; no identities or cross-camera association."""

from __future__ import annotations

import hashlib
import math


def partition_for_image(name: str) -> str:
    """Keep VisDrone images with the same sequence prefix in one partition."""
    sequence = name.split("_")[0]
    return "calibration" if int(hashlib.sha256(sequence.encode()).hexdigest()[:8], 16) % 3 == 0 else "report"


def count_metrics(rows: list[dict], threshold: float) -> dict:
    if not rows:
        raise ValueError("Count evaluation requires images.")
    errors = [sum(score >= threshold for score in row["scores"]) - row["actual"] for row in rows]
    return {
        "images": len(rows), "threshold": threshold,
        "mae": sum(abs(error) for error in errors) / len(errors),
        "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
        "bias": sum(errors) / len(errors),
        "ground_truth_total": sum(row["actual"] for row in rows),
        "predicted_total": sum(row["actual"] + error for row, error in zip(rows, errors)),
    }


def evaluate_counts(rows: list[dict]) -> dict:
    calibration = [row for row in rows if partition_for_image(row["image"]) == "calibration"]
    report = [row for row in rows if partition_for_image(row["image"]) == "report"]
    if not calibration or not report:
        raise ValueError("Need at least one sequence in each calibration/report partition.")
    sweep = [count_metrics(calibration, round(index / 100, 2)) for index in range(5, 66, 5)]
    chosen = min(sweep, key=lambda item: (item["mae"], item["rmse"]))
    return {
        "calibration_sweep": sweep,
        "selected_threshold": chosen["threshold"],
        "calibration": chosen,
        "report": count_metrics(report, chosen["threshold"]),
        "report_at_default_035": count_metrics(report, 0.35),
        "protocol": "Sequence-grouped calibration/report partitions within VisDrone validation. Validation also selects checkpoints; these are development results, not an independent test benchmark.",
    }
