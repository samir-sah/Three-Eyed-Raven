"""Isolated full-validation resolution sweep for anonymous crowd detection.

Never runs tracking, identity embeddings, or automatic checkpoint deployment.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import traceback
from datetime import datetime
from pathlib import Path

from .extended_detection_cli import Status, atomic_json, audit_dataset, evaluate, now, write_report


def valid_sizes(values):
    sizes = [int(value) for value in values.split(",")]
    if not sizes or len(set(sizes)) != len(sizes) or any(size < 320 or size > 2560 or size % 32 for size in sizes):
        raise argparse.ArgumentTypeError("Use unique multiples of 32 between 320 and 2560.")
    return sizes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--name", default="detection_resolution_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sizes", type=valid_sizes, default=valid_sizes("960,1280,1536,1920,2240"))
    parser.add_argument("--tta", action="store_true", help="Evaluate test-time augmentation consistently for detection and counts.")
    args = parser.parse_args()
    if not args.name.replace("_", "").isalnum():
        parser.error("Name must contain only letters, digits, and underscores.")
    root = args.project_root.resolve()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error("Checkpoint does not exist.")
    directory = root / "runs/experiments" / args.name
    directory.mkdir(parents=True, exist_ok=False)
    state = Status(directory)
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this bounded evaluation sweep.")
        torch.set_num_threads(4)
        dataset = root / "data/processed/visdrone_person"
        audit = audit_dataset(dataset)
        atomic_json(directory / "dataset_audit.json", audit)
        data = directory / "dataset.yaml"
        data.write_text(f"path: {json.dumps(dataset.as_posix())}\ntrain: train/images\nval: val/images\nnames:\n  0: person\n", encoding="utf-8")
        state.update(dataset=audit, experimentType="resolution_evaluation", targetMetric="mAP50", target=0.70,
                     checkpoint=str(checkpoint), checkpointSHA256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                     sizes=args.sizes, testTimeAugmentation=args.tta, gpu=torch.cuda.get_device_name(), stages=[])
        evaluations, errors = [], []
        for size in args.sizes:
            name = f"resolution_{size}"
            stage = {"name": name, "imgsz": size, "status": "evaluating", "startedAt": now()}
            state.data["stages"].append(stage)
            state.update(activeStage=name)
            try:
                result = evaluate(checkpoint, name, size, data, dataset, directory, state, augment=args.tta)
                evaluations.append(result)
                stage.update(status="completed", metrics=result["detection"], finishedAt=now())
            except Exception as error:
                stage.update(status="failed", error=str(error), finishedAt=now())
                errors.append({"stage": name, "error": str(error)})
                traceback.print_exc()
            finally:
                gc.collect()
                torch.cuda.empty_cache()
            state.update()
        report = write_report(directory, audit, evaluations, errors, False)
        report.update(target_metric="mAP50", target=0.70,
                      target_reached=any(item["detection"].get("metrics/mAP50(B)", 0) > 0.70 for item in evaluations),
                      checkpoint_sha256=state.data["checkpointSHA256"],
                      test_time_augmentation=args.tta,
                      experiment="One fixed checkpoint and augmentation setting, full validation split, resolution is the only swept setting.")
        atomic_json(directory / "comparison.json", report)
        with (directory / "REPORT.md").open("a", encoding="utf-8") as handle:
            handle.write(f"\nTarget: strictly greater than 70% validation mAP50. Reached: {report['target_reached']}.\n")
        state.update(status="completed_with_errors" if errors else "completed", comparison=report,
                     finishedAt=now(), reportPath=str(directory / "REPORT.md"))
    except BaseException:
        state.update(status="failed", error=traceback.format_exc(), finishedAt=now())
        raise


if __name__ == "__main__":
    main()
