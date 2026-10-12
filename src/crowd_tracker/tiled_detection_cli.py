"""Full-image versus tiled anonymous detection; no tracking or identity features."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

from .count_evaluation import evaluate_counts
from .extended_detection_cli import Status, atomic_json, audit_dataset, clean_metrics, now


def axis_starts(length, tile, overlap=0.2):
    if length <= 0 or tile <= 0 or not 0 <= overlap < 1:
        raise ValueError("Invalid image/tile dimensions or overlap")
    last = max(0, length - tile)
    step = max(1, int(tile * (1 - overlap)))
    return sorted(set([*range(0, last + 1, step), last]))


def tile_windows(width, height, tile):
    return [(x, y, min(x + tile, width), min(y + tile, height))
            for y in axis_starts(height, tile) for x in axis_starts(width, tile)]


def label_boxes(text, width, height):
    boxes = []
    for line in text.splitlines():
        if not line.strip():
            continue
        cls, x, y, w, h = map(float, line.split())
        if cls != 0:
            raise ValueError("Only person labels are supported")
        boxes.append([(x - w / 2) * width, (y - h / 2) * height,
                      (x + w / 2) * width, (y + h / 2) * height])
    return boxes


def evaluate_setting(model, dataset, directory, state, *, tile, size=1536, limit=None):
    import cv2
    import numpy as np
    import torch
    from torchvision.ops import nms
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils.metrics import DetMetrics, box_iou

    name = "whole_image" if not tile else f"tiles_{tile}"
    matcher = DetectionValidator(save_dir=directory / name, args={"plots": False})
    metrics = DetMetrics(names={0: "person"})
    rows, diagnostics = [], []
    buckets = {key: {"ground_truth": 0, "covered": 0} for key in ("small", "medium", "large")}
    images = sorted((dataset / "val/images").glob("*.jpg"))
    if limit:
        images = images[:limit]
    state.update(status="evaluating", activeStage=name, evaluationImagesDone=0, evaluationImagesTotal=len(images))
    started = time.perf_counter()
    calls = 0
    for index, path in enumerate(images):
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Unreadable image: {path}")
        height, width = frame.shape[:2]
        truth = torch.tensor(label_boxes((dataset / "val/labels" / (path.stem + ".txt")).read_text(), width, height), dtype=torch.float32).reshape(-1, 4)
        windows = [(0, 0, width, height)]
        if tile:
            windows += [window for window in tile_windows(width, height, tile) if window != windows[0]]
        predictions = []
        for window_index, (x1, y1, x2, y2) in enumerate(windows):
            result = model.predict(frame[y1:y2, x1:x2], imgsz=size if window_index == 0 else 960,
                                   conf=0.001, iou=0.7, max_det=1000, device=0, classes=[0],
                                   verbose=False, save=False, augment=False)[0]
            boxes = result.boxes.data.detach().cpu().clone()
            calls += 1
            if window_index and len(boxes):
                # Reject boxes cut by internal tile borders; the overlapping
                # tile or the whole-image pass can provide a complete box.
                keep = torch.ones(len(boxes), dtype=torch.bool)
                if x1: keep &= boxes[:, 0] > 2
                if y1: keep &= boxes[:, 1] > 2
                if x2 < width: keep &= boxes[:, 2] < x2 - x1 - 2
                if y2 < height: keep &= boxes[:, 3] < y2 - y1 - 2
                boxes = boxes[keep]
            boxes[:, [0, 2]] += x1
            boxes[:, [1, 3]] += y1
            predictions.append(boxes)
        pred = torch.cat(predictions)
        if tile and len(pred):
            pred = pred[nms(pred[:, :4], pred[:, 4], 0.5)[:1000]]
        p = {"bboxes": pred[:, :4], "cls": pred[:, 5]}
        gt = {"bboxes": truth, "cls": torch.zeros(len(truth))}
        tp = matcher._process_batch(p, gt)["tp"]
        metrics.update_stats({"tp": tp, "conf": pred[:, 4].numpy(), "pred_cls": pred[:, 5].numpy(),
                              "target_cls": gt["cls"].numpy(), "target_img": np.unique(gt["cls"].numpy()), "im_name": path.name})
        scores = pred[:, 4].tolist()
        rows.append({"image": path.name, "actual": len(truth), "scores": scores})
        operating = pred[pred[:, 4] >= 0.2]
        covered = (box_iou(truth, operating[:, :4]).max(1).values >= 0.5) if len(truth) and len(operating) else torch.zeros(len(truth), dtype=torch.bool)
        areas = (truth[:, 2] - truth[:, 0]) * (truth[:, 3] - truth[:, 1])
        for box_index, area in enumerate(areas.tolist()):
            key = "small" if area < 32**2 else "medium" if area < 96**2 else "large"
            buckets[key]["ground_truth"] += 1
            buckets[key]["covered"] += int(covered[box_index])
        diagnostics.append({"image": path.name, "actual": len(truth), "count_at_020": len(operating),
                            "uncovered_gt_at_020": len(truth) - int(covered.sum())})
        if index % 10 == 0 or index + 1 == len(images):
            state.update(evaluationImagesDone=index + 1)
    metrics.process(plot=False)
    elapsed = time.perf_counter() - started
    report = {"name": name, "tile_size": tile, "detection": clean_metrics(metrics.results_dict),
              "images": len(images), "prediction_calls": calls, "elapsed_seconds": elapsed,
              "images_per_second": len(images) / elapsed, "size_coverage_at_conf020": buckets,
              "coverage_note": "GT boxes having any prediction at IoU>=0.5 and confidence>=0.20. Diagnostic coverage, not one-to-one recall. Size thresholds use original-image pixel area.",
              "worst_images": sorted(diagnostics, key=lambda row: row["uncovered_gt_at_020"], reverse=True)[:30]}
    if not limit:
        report["counts"] = evaluate_counts(rows)
    atomic_json(directory / f"{name}.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.name.replace("_", "").isalnum() or not args.checkpoint.is_file():
        parser.error("Provide a valid checkpoint and alphanumeric/underscore experiment name")
    root = Path(__file__).resolve().parents[2]
    directory = root / "runs/experiments" / args.name
    directory.mkdir(parents=True, exist_ok=False)
    state = Status(directory)
    try:
        import torch
        from ultralytics import YOLO
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        torch.set_num_threads(4)
        checkpoint = args.checkpoint.resolve()
        dataset = root / "data/processed/visdrone_person"
        audit = audit_dataset(dataset)
        state.update(dataset=audit, checkpoint=str(checkpoint), checkpointSHA256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), smoke=args.smoke)
        model = YOLO(str(checkpoint))
        reports = [evaluate_setting(model, dataset, directory, state, tile=tile, limit=4 if args.smoke else None) for tile in (0, 640, 960)]
        comparison = {"evaluations": reports, "dataset": audit, "smoke": args.smoke,
                      "target_reached": not args.smoke and any(r["detection"]["metrics/mAP50(B)"] > 0.7 for r in reports),
                      "protocol": "Ultralytics matching/AP on original-image predictions. conf=.001, max_det=1000. Whole image 1536px, tiles 960px, overlap=.20, merge NMS IoU=.50. Predict preprocessing differs from model.val; compare against the whole-image control here. Validation only, no independent test or identity linking."}
        atomic_json(directory / "comparison.json", comparison)
        state.update(status="completed", comparison=comparison, finishedAt=now())
    except BaseException:
        state.update(status="failed", error=traceback.format_exc(), finishedAt=now())
        raise


if __name__ == "__main__":
    main()
