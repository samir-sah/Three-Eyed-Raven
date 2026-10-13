"""Bounded ground-domain pilot; scene-separated anonymous detection, no identities."""
from __future__ import annotations

import argparse
import configparser
import ctypes
import gc
import hashlib
import json
import os
import shutil
import traceback
from pathlib import Path

from .extended_detection_cli import Status, atomic_json, now, fit_stage, audit_dataset
from .ground_detection_cli import parse_annotations, evaluate_model

TRAIN = {"MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN"}
VALIDATION = {"MOT17-10-FRCNN"}


def yolo_labels(boxes, width, height):
    labels = []
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 > x1 and y2 > y1:
            labels.append(f"0 {(x1+x2)/2/width:.8f} {(y1+y2)/2/height:.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}")
    return "\n".join(labels) + ("\n" if labels else "")


def prepare(root, directory):
    manifest = []
    for scene in sorted(TRAIN | VALIDATION):
        source = root / "data/raw/MOT17/train" / scene
        info = configparser.ConfigParser()
        info.read(source / "seqinfo.ini")
        meta = info["Sequence"]
        annotations = parse_annotations((source / "gt/gt.txt").read_text())
        split = "train" if scene in TRAIN else "val"
        images, labels = directory / split / "images", directory / split / "labels"
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        # Approximately two images per second, rather than near-duplicate frames.
        step = max(1, round(int(meta["frameRate"]) / 2))
        for frame in range(1, int(meta["seqLength"]) + 1, step):
            image = source / meta["imDir"] / f"{frame:06d}{meta['imExt']}"
            name = f"{scene}_{frame:06d}"
            shutil.copy2(image, images / f"{name}.jpg")
            text = yolo_labels(annotations[frame]["truth"], int(meta["imWidth"]), int(meta["imHeight"]))
            (labels / f"{name}.txt").write_text(text)
            manifest.append({"scene": scene, "frame": frame, "split": split,
                             "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                             "label_sha256": hashlib.sha256(text.encode()).hexdigest()})
    atomic_json(directory / "manifest.json", {"images": manifest, "scope": "Anonymous class-1 marked pedestrians; original identity column discarded.",
        "limitations": "Only three training scenes and one validation scene. Correlated frames remain within each scene. Ignored regions are not supported by YOLO training loss; excluded categories are treated as background. Full-body occluded boxes remain targets."})
    return audit_dataset(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--hours", type=float, default=.6)
    args = parser.parse_args()
    if not args.name.replace("_", "").isalnum() or not 0 < args.hours <= 1:
        parser.error("Use a safe name and at most one training hour")
    root = Path(__file__).resolve().parents[2]
    directory = root / "runs/experiments" / args.name
    directory.mkdir(exist_ok=False)
    state = Status(directory)
    sleep_inhibited = False
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the bounded laptop pilot")
        torch.set_num_threads(4)
        if os.name == "nt":
            sleep_inhibited = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        dataset = directory / "dataset"
        audit = prepare(root, dataset)
        data = directory / "dataset.yaml"
        data.write_text(f"path: {json.dumps(dataset.as_posix())}\ntrain: train/images\nval: val/images\nnames:\n  0: person\n")
        state.update(dataset=audit, budgetHours=args.hours, architecture="YOLO11s ground-domain pilot", gpu=torch.cuda.get_device_name(0),
                     datasetDescription="Scene-separated MOT17 development subset, sampled about twice per second; not full-data training", idleSleepInhibited=sleep_inhibited)
        atomic_json(directory.parent / "active.json", {"name": args.name})
        checkpoint = fit_stage(root / "yolo11s.pt", directory, data, state, size=960, batch=1, hours=args.hours,
                               smoke=False, learning_rate=.0001, workers=0, stage_name="ground_person_960")
        gc.collect()
        torch.cuda.empty_cache()
        previous = json.loads((root / "runs/experiments/ground_detection_20261013_corrected/manifest.json").read_text())
        # Threshold selection uses only the validation scene, never training scenes.
        manifest = [{**row, "partition": "train" if row["sequence"] in TRAIN else row["partition"]} for row in previous["images"]]
        atomic_json(directory / "evaluation_manifest.json", {**previous, "images": manifest})
        results = []
        for label, weights in (("current_coco_yolo11s_960", root / "yolo11s.pt"), ("ground_pilot_960", checkpoint)):
            torch.cuda.reset_peak_memory_stats()
            results.append(evaluate_model(weights, label, manifest, directory, state, 960))
        # Adapter keeps the training monitor's existing result schema stable.
        evaluations = [{"name": item["name"], "detection": {"metrics/mAP50(B)": item["report_detection"]["map50"], "metrics/mAP50-95(B)": item["report_detection"]["map5095"]},
                        "counts": {"selected_threshold": .35, "report": {"mae": item["report_default"]["count_mae"], "rmse": item["report_default"]["count_rmse"], "threshold": .35}}} for item in results]
        comparison = {"evaluations": evaluations, "detailed_evaluations": results, "finishedAt": now(),
            "limitation": "Exploratory MOT17 development audit: 78 report images from three scenes, already inspected by this project, not independent test data. Three other scenes train the pilot; scene 10 selects checkpoints. Operating threshold stays .35 for both models. No deployment, tracking, or cross-camera linking.",
            "recommended_by_validation_map": max(evaluations, key=lambda item: item["detection"]["metrics/mAP50(B)"])["name"], "errors": []}
        atomic_json(directory / "comparison.json", comparison)
        lines = ["# Ground-domain laptop pilot", "", comparison["limitation"], "", "| Model | Report mAP50 | Recall at .35 | Precision at .35 | Count MAE |", "|---|---|---|---|---|"]
        for item in results:
            metric = item["report_default"]
            lines.append(f"| {item['name']} | {item['report_detection']['map50']:.2%} | {metric['recall']:.2%} | {metric['precision']:.2%} | {metric['count_mae']:.2f} |")
        lines += ["", "Checkpoints remain isolated. Review per-scene precision/recall and count errors before any anonymous-only promotion. Validation scores from training are not the custom report mAP scores above."]
        (directory / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        state.update(status="completed", finishedAt=now(), comparison=comparison, reportPath=str(directory / "REPORT.md"))
    except BaseException:
        state.update(status="failed", error=traceback.format_exc(), finishedAt=now())
        raise
    finally:
        if sleep_inhibited:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
