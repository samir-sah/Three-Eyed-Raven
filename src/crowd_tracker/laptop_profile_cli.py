"""Matched anonymous ground-detector profiles; no training or identity processing."""
import argparse
import json
import traceback
from pathlib import Path

from .extended_detection_cli import Status, atomic_json, now
from .ground_detection_cli import evaluate_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if not args.name.replace("_", "").isalnum():
        parser.error("Unsafe experiment name")
    root = Path(__file__).resolve().parents[2]
    directory = root / "runs/experiments" / args.name
    directory.mkdir(exist_ok=False)
    state = Status(directory)
    try:
        import torch
        torch.set_num_threads(4)
        previous = json.loads((root / "runs/experiments/ground_detection_20261013_corrected/manifest.json").read_text())
        state.update(images=len(previous["images"]), scenes=previous["scenes"])
        atomic_json(directory / "manifest.json", previous)
        evaluations = []
        for size in (640, 960):
            for architecture in ("yolo11n", "yolo11s"):
                torch.cuda.reset_peak_memory_stats()
                evaluations.append(evaluate_model(root / f"{architecture}.pt", f"coco_{architecture}_{size}",
                                                  previous["images"], directory, state, size))
        comparison = {"finishedAt": now(), "protocol": {**previous["protocol"], "inference": {"resolutions": [640, 960], "minimum_confidence": .001, "operating_confidence": .35, "nms_iou": .5, "max_det": 300, "augmentation": False}}, "evaluations": evaluations,
                      "limitations": "Development scenes, not independent test data. Model selection is exploratory. No tracking, linking or deployment."}
        atomic_json(directory / "comparison.json", comparison)
        lines = ["# Laptop ground detector profiles", "", comparison["limitations"], "",
                 "Same 215 images and annotation rules as the corrected ground audit. 78 report images; 137 calibration images. Confidence .35; AP integrates confidence scores.", "",
                 "| Model | mAP50 | Recall | Precision | Count MAE | Decode + inference FPS |", "|---|---|---|---|---|---|"]
        for result in evaluations:
            metric = result["report_default"]
            lines.append(f"| {result['name']} | {result['report_detection']['map50']:.2%} | {metric['recall']:.2%} | {metric['precision']:.2%} | {metric['count_mae']:.2f} | {result['speed']['decode_predict_fps']:.1f} |")
        (directory / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        state.update(status="completed", finishedAt=now())
        atomic_json(root / "runs/experiments/ground_active.json", {"name": args.name})
    except BaseException:
        state.update(status="failed", error=traceback.format_exc(), finishedAt=now())
        raise


if __name__ == "__main__":
    main()
