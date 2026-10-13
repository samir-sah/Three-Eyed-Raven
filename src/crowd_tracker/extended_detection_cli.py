"""Durable, time-budgeted full-data training and aggregate-count evaluation.

This experiment never runs tracking, person embeddings, or cross-camera matching.
Checkpoints stay isolated until their comparative evaluation has been reviewed.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .count_evaluation import evaluate_counts


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.1)


class Status:
    def __init__(self, directory: Path):
        self.path = directory / "status.json"
        self.data = {"startedAt": now(), "pid": os.getpid(), "status": "preflight", "stages": []}
        self.update()

    def update(self, **values):
        self.data.update(values)
        self.data["updatedAt"] = now()
        atomic_json(self.path, self.data)


def audit_dataset(root: Path) -> dict:
    result = {}
    all_names = []
    digest = hashlib.sha256()
    for split in ("train", "val"):
        images = sorted((root / split / "images").glob("*.jpg"))
        if not images:
            raise ValueError(f"No {split} images in {root}")
        names, boxes = set(), 0
        for image in images:
            names.add(image.stem)
            label = root / split / "labels" / f"{image.stem}.txt"
            text = label.read_text(encoding="utf-8")
            digest.update(f"{split}/{image.name}\n{text}".encode())
            for line in text.splitlines():
                if not line.strip():
                    continue
                values = [float(value) for value in line.split()]
                if len(values) != 5 or values[0] != 0 or not all(math.isfinite(v) and 0 <= v <= 1 for v in values[1:]) or min(values[3:]) <= 0:
                    raise ValueError(f"Invalid person label in {label}: {line}")
                boxes += 1
        all_names.append(names)
        result[split] = {"images": len(images), "boxes": boxes}
    overlap = all_names[0] & all_names[1]
    if overlap:
        raise ValueError(f"Train/validation filenames overlap: {len(overlap)}")
    result["annotation_sha256"] = digest.hexdigest()
    result["filename_overlap"] = 0
    result["independent_test_available"] = False
    return result


def clean_metrics(metrics) -> dict:
    return {key: float(value) for key, value in metrics.items() if math.isfinite(float(value))}


def fit_stage(checkpoint: Path, directory: Path, data: Path, state: Status, *, size: int, batch: int, hours: float, smoke: bool, learning_rate: float | None = None, warmup_epochs: float = 1, workers: int = 2, stage_name: str | None = None) -> Path:
    import torch
    from ultralytics import YOLO

    name = stage_name or f"visdrone_person_{size}"
    folder = directory / name
    current = {"name": name, "status": "training", "imgsz": size, "budgetHours": hours, "checkpoint_source": str(checkpoint), "startedAt": now()}
    state.data["stages"].append(current)
    model = YOLO(str(checkpoint))
    last_update = 0.0
    batch_index = 0

    def epoch_start(trainer):
        nonlocal batch_index
        batch_index = 0
        state.update(status="training", activeStage=name, epoch=trainer.epoch + 1, totalEpochs=trainer.epochs, batchesDone=0, batchesTotal=len(trainer.train_loader))

    def progress(trainer):
        nonlocal last_update, batch_index
        batch_index += 1
        if time.monotonic() - last_update >= 15:
            last_update = time.monotonic()
            state.update(batchesDone=batch_index, gpuAllocatedMB=round(torch.cuda.memory_allocated() / 2**20))

    def epoch_saved(trainer):
        current.update(epoch=trainer.epoch + 1, metrics=clean_metrics(trainer.metrics), bestCheckpoint=str(trainer.best), lastCheckpoint=str(trainer.last))
        state.update(metrics=current["metrics"], checkpointSavedAt=now())

    model.add_callback("on_train_epoch_start", epoch_start)
    model.add_callback("on_train_batch_end", progress)
    model.add_callback("on_model_save", epoch_saved)
    model.add_callback("on_train_epoch_end", lambda trainer: state.update(status="validating", batchesDone=batch_index))
    try:
        model.train(
            data=str(data), project=str(directory), name=name, exist_ok=False,
            epochs=1 if smoke else 150, time=None if smoke else hours,
            imgsz=size, batch=batch, device=0, workers=workers,
            optimizer="AdamW", lr0=learning_rate if learning_rate is not None else (0.0001 if size == 640 else 0.00005),
            lrf=0.05, cos_lr=True, warmup_epochs=warmup_epochs, patience=20,
            save=True, save_period=10, seed=42, deterministic=True,
            cache=False, amp=True, plots=False, close_mosaic=0,
            mosaic=0.5, mixup=0.0, max_det=1000, fraction=0.005 if smoke else 1.0,
        )
        best = Path(model.trainer.best)
        if not best.is_file():
            raise RuntimeError("Training returned without a best checkpoint.")
        current.update(status="completed", finishedAt=now(), bestCheckpoint=str(best))
        state.update(status="between_stages")
        return best
    except Exception:
        current.update(status="failed", error=traceback.format_exc(), finishedAt=now())
        state.update(status="stage_failed")
        raise
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def evaluate(checkpoint: Path, name: str, size: int, data: Path, dataset: Path, directory: Path, state: Status, *, augment: bool = False, workers: int = 2) -> dict:
    import torch
    from ultralytics import YOLO

    state.update(status="evaluating", activeStage=name, evaluationImagesDone=0)
    model = YOLO(str(checkpoint))
    try:
        metrics = model.val(data=str(data), split="val", imgsz=size, batch=1, device=0,
                            workers=workers, plots=False, save_json=False, conf=0.001, max_det=1000, augment=augment,
                            project=str(directory / "evaluation"), name=name, exist_ok=False)
        images = sorted((dataset / "val" / "images").glob("*.jpg"))
        rows = []
        started = time.perf_counter()
        # A list can become a single tensor batch despite batch=1 in some
        # Ultralytics versions. One path per call bounds GPU and host memory.
        for start in range(len(images)):
            for result in model.predict(source=str(images[start]), stream=True,
                    imgsz=size, batch=1, device=0, conf=0.05, iou=0.7, max_det=1000,
                    classes=[0], verbose=False, save=False, augment=augment):
                image = Path(result.path)
                labels = (dataset / "val" / "labels" / f"{image.stem}.txt").read_text().splitlines()
                rows.append({"image": image.name, "actual": sum(bool(line.strip()) for line in labels),
                             "scores": result.boxes.conf.cpu().tolist()})
            state.update(evaluationImagesDone=len(rows), evaluationImagesTotal=len(images))
        elapsed = time.perf_counter() - started
        report = {"name": name, "checkpoint": str(checkpoint), "imgsz": size, "test_time_augmentation": augment,
                  "detection": clean_metrics(metrics.results_dict), "counts": evaluate_counts(rows),
                  "validation_inference_seconds": elapsed, "images_per_second": len(images) / elapsed}
        atomic_json(directory / f"{name}_evaluation.json", report)
        return report
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def write_report(directory: Path, audit: dict, evaluations: list[dict], errors: list[dict], smoke: bool) -> dict:
    winner = max(evaluations, key=lambda item: item["detection"].get("metrics/mAP50-95(B)", -1)) if evaluations else None
    payload = {"finishedAt": now(), "smoke_test": smoke, "dataset": audit, "evaluations": evaluations,
               "errors": errors, "recommended_by_validation_map": winner["name"] if winner else None,
               "scope": "Anonymous image-level crowd detection/counting only. No tracking or cross-camera identification.",
               "limitation": "Model selection and counting analysis use validation data. No independent test dataset was available. Candidate checkpoints are not automatically deployed."}
    atomic_json(directory / "comparison.json", payload)
    lines = ["# Extended crowd detection experiment", "", payload["scope"], "", payload["limitation"], "",
             "| Model | Resolution | mAP50 | mAP50–95 | Count MAE | Count RMSE | Threshold |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for item in evaluations:
        det, counts = item["detection"], item["counts"]["report"]
        lines.append(f"| {item['name']} | {item['imgsz']} | {det.get('metrics/mAP50(B)', 0):.4f} | {det.get('metrics/mAP50-95(B)', 0):.4f} | {counts['mae']:.3f} | {counts['rmse']:.3f} | {counts['threshold']:.2f} |")
    lines += ["", f"Validation mAP recommendation: {payload['recommended_by_validation_map']}", "",
              "Count threshold is fitted on the calibration partition; count errors are computed on the separate report partition. Both partitions belong to the validation dataset.", ""]
    if errors:
        lines += ["## Failed stages", "", *[f"- {error['stage']}: {error['error']}" for error in errors]]
    (directory / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--name", default="extended_detection_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--hours", type=float, default=5.0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not 0 < args.hours <= 8 or not args.name.replace("_", "").isalnum():
        parser.error("hours must be in (0, 8]; name must contain only letters, digits, underscores")
    root = Path(args.project_root).resolve()
    directory = root / "runs" / "experiments" / args.name
    directory.mkdir(parents=True, exist_ok=False)
    state = Status(directory)
    prevent_sleep = False
    try:
        if os.name == "nt":
            prevent_sleep = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        import torch
        import ultralytics
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this unattended GPU experiment.")
        if shutil.disk_usage(root).free < 8 * 2**30:
            raise RuntimeError("Need at least 8 GiB free space for checkpoints and evaluation.")
        torch.set_num_threads(4)
        dataset = root / "data" / "processed" / "visdrone_person"
        audit = audit_dataset(dataset)
        atomic_json(directory / "dataset_audit.json", audit)
        data = directory / "dataset.yaml"
        data.write_text(f"path: {json.dumps(dataset.as_posix())}\ntrain: train/images\nval: val/images\nnames:\n  0: person\n", encoding="utf-8")
        baseline = root / "runs/detect/runs/detect/visdrone_person_50e/weights/best.pt"
        if not baseline.is_file():
            raise FileNotFoundError(baseline)
        state.update(dataset=audit, budgetHours=args.hours, smoke=args.smoke,
                     torch=torch.__version__, ultralytics=ultralytics.__version__,
                     gpu=torch.cuda.get_device_name(), idleSleepInhibited=prevent_sleep,
                     baselineCheckpoint=str(baseline))
        if not args.smoke:
            atomic_json(directory.parent / "active.json", {"directory": str(directory), "name": args.name})
        checkpoints = [("baseline_640", baseline, 640)]
        if not args.smoke:
            checkpoints.append(("baseline_960", baseline, 960))
        errors = []
        source = baseline
        stages = [(640, 4, args.hours * 0.6)] if args.smoke else [(640, 4, args.hours * 0.6), (960, 2, args.hours * 0.4)]
        for size, batch, hours in stages:
            try:
                source = fit_stage(source, directory, data, state, size=size, batch=batch, hours=hours, smoke=args.smoke)
                checkpoints.append((f"candidate_{size}", source, size))
            except Exception as error:
                errors.append({"stage": f"training_{size}", "error": str(error)})
                traceback.print_exc()
        evaluations = []
        for name, checkpoint, size in checkpoints:
            try:
                evaluations.append(evaluate(checkpoint, name, size, data, dataset, directory, state))
            except Exception as error:
                errors.append({"stage": f"evaluation_{name}", "error": str(error)})
                traceback.print_exc()
        report = write_report(directory, audit, evaluations, errors, args.smoke)
        state.update(status="completed_with_errors" if errors else "completed", finishedAt=now(),
                     comparison=report, reportPath=str(directory / "REPORT.md"))
    except BaseException:
        state.update(status="failed", error=traceback.format_exc(), finishedAt=now())
        raise
    finally:
        if prevent_sleep:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
