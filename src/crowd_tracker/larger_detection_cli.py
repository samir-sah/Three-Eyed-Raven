"""Time-bounded YOLO11s training for anonymous person detection only."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import traceback
from pathlib import Path

from .extended_detection_cli import Status, atomic_json, audit_dataset, evaluate, fit_stage, now, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--hours", type=float, default=3)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=0, help="Zero avoids RAM-heavy Windows data-loader child processes")
    args = parser.parse_args()
    if not args.name.replace("_", "").isalnum() or not 0 < args.hours <= 6 or args.workers < 0:
        parser.error("Use a unique alphanumeric/underscore name and a budget in (0,6] hours")
    root = Path(__file__).resolve().parents[2]
    pretrained = root / "yolo11s.pt"
    baseline = root / "runs/experiments/extended_detection_20261012/visdrone_person_960/weights/best.pt"
    if not pretrained.is_file() or not baseline.is_file():
        parser.error("Both official yolo11s.pt and the previous trained nano checkpoint must exist")
    directory = root / "runs/experiments" / args.name
    directory.mkdir(parents=True, exist_ok=False)
    state = Status(directory)
    sleep_inhibited = False
    try:
        import torch
        import ultralytics
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        if shutil.disk_usage(root).free < 8 * 2**30:
            raise RuntimeError("At least 8 GiB free space required")
        if os.name == "nt":
            sleep_inhibited = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        torch.set_num_threads(4)
        dataset = root / "data/processed/visdrone_person"
        audit = audit_dataset(dataset)
        atomic_json(directory / "dataset_audit.json", audit)
        data = directory / "dataset.yaml"
        data.write_text(f"path: {json.dumps(dataset.as_posix())}\ntrain: train/images\nval: val/images\nnames:\n  0: person\n", encoding="utf-8")
        state.update(dataset=audit, budgetHours=args.hours, smoke=args.smoke, architecture="YOLO11s", workers=args.workers,
                     gpu=torch.cuda.get_device_name(), torch=torch.__version__, ultralytics=ultralytics.__version__,
                     pretrainedSHA256=hashlib.sha256(pretrained.read_bytes()).hexdigest(),
                     targetMetric="mAP50", target=0.70, idleSleepInhibited=sleep_inhibited,
                     scope="Anonymous image-level detection and aggregate counts only; no deployment")
        if not args.smoke:
            atomic_json(directory.parent / "active.json", {"directory": str(directory), "name": args.name})
        best = fit_stage(pretrained, directory, data, state, size=960, batch=1,
                         hours=args.hours, smoke=args.smoke, learning_rate=0.001, warmup_epochs=3, workers=args.workers)
        if args.smoke:
            state.update(status="completed", finishedAt=now(), note="Infrastructure smoke only; 0.5% training data, not model quality evidence")
            return
        evaluations, errors = [], []
        for name, checkpoint, size, augment in [
            ("nano_960", baseline, 960, False), ("small_960", best, 960, False),
            ("nano_1536", baseline, 1536, False), ("small_1536", best, 1536, False),
            ("small_1536_tta", best, 1536, True),
        ]:
            try:
                evaluations.append(evaluate(checkpoint, name, size, data, dataset, directory, state, augment=augment, workers=args.workers))
            except Exception as error:
                errors.append({"stage": name, "error": str(error)})
                traceback.print_exc()
        report = write_report(directory, audit, evaluations, errors, False)
        report.update(target_metric="mAP50", target=0.70,
                      target_reached=any(r["detection"]["metrics/mAP50(B)"] > .70 for r in evaluations),
                      design="YOLO11s initialized from official COCO weights; compare to trained YOLO11n at matched inference resolution. Three-hour budget is exploratory, not proof of convergence.")
        atomic_json(directory / "comparison.json", report)
        state.update(status="completed_with_errors" if errors else "completed", comparison=report,
                     finishedAt=now(), reportPath=str(directory / "REPORT.md"))
    except BaseException:
        state.update(status="failed", error=traceback.format_exc(), finishedAt=now())
        raise
    finally:
        if sleep_inhibited:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
