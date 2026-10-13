"""Reproducible anonymous image detection/count audit on sampled MOT17 scenes."""
from __future__ import annotations

import argparse
import configparser
import csv
import gc
import hashlib
import json
import math
import time
import traceback
from collections import defaultdict
from pathlib import Path

from .extended_detection_cli import Status, atomic_json, now

CALIBRATION = {"MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-10-FRCNN"}
REPORT = {"MOT17-09-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"}
THRESHOLDS = [round(value / 100, 2) for value in range(5, 66, 5)]


def parse_annotations(text):
    """Discard identity column; retain marked pedestrians and ignored non-targets."""
    frames = defaultdict(lambda: {"truth": [], "ignore": [], "visibility": []})
    for row in csv.reader(text.splitlines()):
        if not row:
            continue
        if len(row) != 9:
            raise ValueError("Expected nine MOT17 annotation columns")
        frame = int(row[0])
        x, y, w, h, marked, category, visibility = map(float, row[2:])
        if frame < 1 or not all(math.isfinite(v) for v in (x, y, w, h, marked, category, visibility)) or w <= 0 or h <= 0:
            raise ValueError("Invalid MOT17 annotation")
        # MOTChallenge boxes are 1-based; detector/OpenCV coordinates are 0-based.
        box = [x - 1, y - 1, x - 1 + w, y - 1 + h]
        if marked > 0 and category == 1:
            frames[frame]["truth"].append(box)
            frames[frame]["visibility"].append(visibility)
        else:
            frames[frame]["ignore"].append(box)
    return frames


def make_manifest(root, smoke=False):
    rows, scenes = [], []
    for name in sorted(CALIBRATION | REPORT):
        sequence = root / name
        info = configparser.ConfigParser()
        info.read(sequence / "seqinfo.ini")
        meta = info["Sequence"]
        fps, length = int(meta["frameRate"]), int(meta["seqLength"])
        if fps <= 0 or length <= 0:
            raise ValueError("Invalid sequence frame rate/length")
        text = (sequence / "gt/gt.txt").read_text(encoding="utf-8")
        labels = parse_annotations(text)
        frame_numbers = list(range(1, length + 1, fps))
        if smoke:
            frame_numbers = frame_numbers[:1]
        partition = "calibration" if name in CALIBRATION else "report"
        for frame in frame_numbers:
            image = sequence / meta["imDir"] / f"{frame:06d}{meta['imExt']}"
            if not image.is_file():
                raise FileNotFoundError(image)
            rows.append({"image": str(image), "sequence": name, "frame": frame,
                         "timestamp_seconds": (frame - 1) / fps, "partition": partition,
                         "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                         **labels[frame]})
        scenes.append({"sequence": name, "partition": partition, "sampled_images": len(frame_numbers),
                       "source_fps": fps, "source_frames": length,
                       "annotation_sha256": hashlib.sha256(text.encode()).hexdigest()})
    return rows, scenes


def match_frame(truth, ignored, predictions, alpha=0.5):
    """Confidence-ordered one-to-one IoU matching; unmatched ignore overlap is neutral."""
    import numpy as np
    truth = np.asarray(truth, dtype=float).reshape(-1, 4)
    ignored = np.asarray(ignored, dtype=float).reshape(-1, 4)
    predictions = np.asarray(predictions, dtype=float).reshape(-1, 5)
    if len(predictions):
        predictions = predictions[np.argsort(-predictions[:, 4], kind="stable")]
    pred_areas = np.maximum(0, predictions[:, 2:4] - predictions[:, :2]).prod(1)
    intersections = np.maximum(0, np.minimum(predictions[:, None, 2:4], truth[None, :, 2:4]) -
                                np.maximum(predictions[:, None, :2], truth[None, :, :2])).prod(2)
    unions = pred_areas[:, None] + np.maximum(0, truth[:, 2:4] - truth[:, :2]).prod(1)[None, :] - intersections
    overlaps = np.divide(intersections, unions, out=np.zeros_like(intersections), where=unions > 0)
    ignore_intersections = np.maximum(0, np.minimum(predictions[:, None, 2:4], ignored[None, :, 2:4]) -
                                       np.maximum(predictions[:, None, :2], ignored[None, :, :2])).prod(2)
    ignore_overlaps = np.divide(ignore_intersections, pred_areas[:, None], out=np.zeros_like(ignore_intersections), where=pred_areas[:, None] > 0)
    available = np.ones(len(truth), dtype=bool)
    flags, neutral, indices = [], [], []
    for pred_index, box in enumerate(predictions):
        eligible = np.where(available & (overlaps[pred_index] >= alpha))[0]
        index = int(eligible[np.argmax(overlaps[pred_index, eligible])]) if len(eligible) else -1
        if index >= 0:
            available[index] = False
        ignore = bool(index < 0 and (ignore_overlaps[pred_index] >= 0.5).any())
        flags.append(index >= 0)
        neutral.append(ignore)
        indices.append(index)
    return predictions, np.asarray(flags, dtype=bool), np.asarray(neutral, dtype=bool), indices


def operating_metrics(rows, threshold):
    tp = fp = fn = ignored = raw_total = 0
    errors = []
    misses = []
    sizes = {key: {"truth": 0, "matched": 0} for key in ("small", "medium", "large")}
    for row in rows:
        predictions = [p for p in row["predictions"] if p[4] >= threshold]
        _, hits, neutral, indices = match_frame(row["truth"], row["ignore"], predictions)
        matched = int(hits.sum())
        false = len(predictions) - matched - int(neutral.sum())
        actual = len(row["truth"])
        tp += matched
        fp += false
        fn += actual - matched
        ignored += int(neutral.sum())
        raw_total += len(predictions)
        errors.append(matched + false - actual)
        found = set(indices) - {-1}
        for index, (x1, y1, x2, y2) in enumerate(row["truth"]):
            area = (x2 - x1) * (y2 - y1)
            key = "small" if area < 32**2 else "medium" if area < 96**2 else "large"
            sizes[key]["truth"] += 1
            sizes[key]["matched"] += int(index in found)
        misses.append({"sequence": row["sequence"], "frame": row["frame"], "actual": actual,
                       "count": matched + false, "missed": actual - matched, "false_positives": false})
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return {"images": len(rows), "threshold": threshold, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
            "true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "ignored_predictions": ignored, "raw_predicted_total": raw_total,
            "count_mae": sum(map(abs, errors)) / len(errors),
            "count_rmse": math.sqrt(sum(e * e for e in errors) / len(errors)),
            "count_bias": sum(errors) / len(errors), "size_recall": sizes,
            "worst_frames": sorted(misses, key=lambda item: item["missed"], reverse=True)[:10]}


def detection_ap(rows):
    import numpy as np
    from ultralytics.utils.metrics import compute_ap
    targets = sum(len(row["truth"]) for row in rows)
    aps = []
    for alpha in np.arange(0.5, 0.96, 0.05):
        scores, correct = [], []
        for row in rows:
            predictions, hits, neutral, _ = match_frame(row["truth"], row["ignore"], row["predictions"], round(float(alpha), 2))
            scores.extend(predictions[~neutral, 4].tolist())
            correct.extend(hits[~neutral].tolist())
        order = np.argsort(-np.asarray(scores), kind="stable")
        hits = np.asarray(correct, dtype=float)[order]
        if not targets or not len(hits):
            aps.append(0.0)
            continue
        tp, fp = hits.cumsum(), (1 - hits).cumsum()
        aps.append(float(compute_ap(tp / targets, tp / np.maximum(tp + fp, 1))[0]))
    return {"map50": aps[0], "map5095": float(np.mean(aps)), "ground_truth_boxes": targets}


def evaluate_model(checkpoint, label, manifest, directory, state, size):
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    model = YOLO(str(checkpoint))
    if model.names.get(0) not in ("person", "pedestrian", "people"):
        raise ValueError(f"Person class 0 is not valid for {checkpoint}: {model.names}")
    state.update(status="evaluating", activeStage=label, evaluationImagesDone=0, evaluationImagesTotal=len(manifest))
    rows, latencies = [], []
    # Warm up separately; reported latency covers decode plus predict, not evaluation.
    model.predict(np.zeros((size, size, 3), dtype=np.uint8), imgsz=size, device=0, verbose=False)
    with (directory / f"{label}_predictions.jsonl").open("w", encoding="utf-8") as output:
        for index, item in enumerate(manifest):
            torch.cuda.synchronize()
            started = time.perf_counter()
            frame = cv2.imread(item["image"])
            if frame is None:
                raise ValueError(f"Unreadable image: {item['image']}")
            prediction = model.predict(frame, imgsz=size, conf=0.001, iou=0.5, max_det=300,
                                       classes=[0], device=0, augment=False, verbose=False, save=False)[0]
            boxes = prediction.boxes.data[:, :5].detach().cpu().tolist()
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - started) * 1000)
            row = {**item, "predictions": boxes}
            rows.append(row)
            output.write(json.dumps({"sequence": item["sequence"], "frame": item["frame"],
                                     "timestamp_seconds": item["timestamp_seconds"], "boxes_scores": boxes}) + "\n")
            if index % 10 == 0 or index + 1 == len(manifest):
                state.update(evaluationImagesDone=index + 1)
    calibration = [row for row in rows if row["partition"] == "calibration"]
    report = [row for row in rows if row["partition"] == "report"]
    sweep = [operating_metrics(calibration, threshold) for threshold in THRESHOLDS]
    selected = max(sweep, key=lambda metrics: (metrics["f1"], metrics["precision"]))["threshold"]
    result = {"name": label, "checkpoint": str(checkpoint),
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "resolution": size, "report_detection": detection_ap(report),
              "report_default": operating_metrics(report, 0.35),
              "selected_threshold": selected, "report_tuned": operating_metrics(report, selected),
              "calibration_sweep": sweep, "all_default": operating_metrics(rows, 0.35),
              "per_sequence_default": {name: operating_metrics([row for row in rows if row["sequence"] == name], 0.35)
                                       for name in sorted(CALIBRATION | REPORT)},
              "speed": {"mean_decode_predict_ms": float(np.mean(latencies)),
                        "p95_decode_predict_ms": float(np.percentile(latencies, 95)),
                        "decode_predict_fps": 1000 / float(np.mean(latencies)),
                        "peak_allocated_vram_mb": torch.cuda.max_memory_allocated() / 2**20}}
    atomic_json(directory / f"{label}.json", result)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.name.replace("_", "").isalnum() or args.imgsz < 320 or args.imgsz > 960:
        parser.error("Use an alphanumeric/underscore name and laptop image size 320–960")
    root = Path(__file__).resolve().parents[2]
    directory = root / "runs/experiments" / args.name
    directory.mkdir(parents=True, exist_ok=False)
    state = Status(directory)
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this laptop audit")
        torch.set_num_threads(4)
        models = {
            "coco_yolo11n": root / "yolo11n.pt",
            "historical_visdrone_50e": root / "runs/detect/runs/detect/visdrone_person_50e/weights/best.pt",
            "extended_visdrone_nano": root / "runs/experiments/extended_detection_20261012/visdrone_person_960/weights/best.pt",
            "experimental_visdrone_small": root / "runs/experiments/larger_detection_20261012_w0/visdrone_person_960/weights/best.pt",
        }
        for checkpoint in models.values():
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
        manifest, scenes = make_manifest(root / "data/raw/MOT17/train", args.smoke)
        protocol = {
            "dataset": "MOT17 public training sequences; development audit, not official MOT benchmark or independent final test",
            "sampling": "First frame and every source-fps frames thereafter (approximately one frame/second)",
            "definition": "Marked class-1 pedestrians, including occluded annotations. MOT 1-based boxes are converted to 0-based detector coordinates. Counts describe the annotated target scope, not every visible human category.",
            "matching": "Confidence-ordered greedy one-to-one IoU matching. Non-target/unmarked boxes neutralize unmatched predictions at prediction-area overlap >=0.5; valid targets take precedence. Custom detection protocol, not official TrackEval.",
            "ap": "Single-person-class AP at IoU .50:.05:.95, using Ultralytics 101-point AP integration. Ignore matching recomputed at each IoU.",
            "inference": {"imgsz": args.imgsz, "minimum_confidence": 0.001, "operating_confidence": 0.35,
                          "nms_iou": 0.5, "max_det": 300, "augmentation": False, "class": 0},
            "partitions": {"calibration": sorted(CALIBRATION), "report": sorted(REPORT)},
            "threshold_selection": "Max calibration F1 over .05:.05:.65, ties by precision; then fixed for report scenes",
            "limitations": "Report scenes are held apart from threshold tuning but MOT17 has already been inspected by the project. No proof of performance on uploaded videos; no confidence intervals with this small scene sample.",
            "scope": "Anonymous image-level boxes and aggregate counts only; no identity column, tracking, embeddings or deployment",
        }
        atomic_json(directory / "manifest.json", {"scenes": scenes, "images": manifest, "protocol": protocol})
        state.update(smoke=args.smoke, images=len(manifest), scenes=scenes, protocol=protocol,
                     architecture="Four existing detector checkpoints", gpu=torch.cuda.get_device_name(0))
        if not args.smoke:
            atomic_json(root / "runs/experiments/ground_active.json", {"name": args.name})
        evaluations = []
        for label, checkpoint in models.items():
            torch.cuda.reset_peak_memory_stats()
            evaluations.append(evaluate_model(checkpoint, label, manifest, directory, state, args.imgsz))
        comparison = {"finishedAt": now(), "smoke": args.smoke, "images": len(manifest),
                      "scenes": scenes, "protocol": protocol, "evaluations": evaluations}
        atomic_json(directory / "comparison.json", comparison)
        lines = ["# Ground-camera detection comparison", "", protocol["dataset"], "", protocol["limitations"], "",
                 "All models use identical sampled frames and inference settings. No tracking or cross-camera matching.", "",
                 "| Model | Report mAP50 | Recall at .35 | Precision at .35 | Count MAE at .35 | Tuned threshold | Tuned recall | Tuned count MAE |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for result in evaluations:
            default, tuned = result["report_default"], result["report_tuned"]
            lines.append(f"| {result['name']} | {result['report_detection']['map50']:.2%} | {default['recall']:.2%} | {default['precision']:.2%} | {default['count_mae']:.2f} | {result['selected_threshold']:.2f} | {tuned['recall']:.2%} | {tuned['count_mae']:.2f} |")
        lines += ["", "Counts exclude neutralized detections on annotated non-target regions. Raw totals and per-scene failures are retained in JSON.", "", "## Protocol", "", *[f"- {key}: {value}" for key, value in protocol.items()]]
        (directory / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        state.update(status="completed", finishedAt=now(), reportPath=str(directory / "REPORT.md"))
    except BaseException:
        state.update(status="failed", error=traceback.format_exc(), finishedAt=now())
        raise


if __name__ == "__main__":
    main()
