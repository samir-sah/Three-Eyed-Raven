"""Calibration-first anonymous ground detector search with a frozen report pass."""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from .detection_profiles import DetectionProfile, predict_boxes
from .extended_detection_cli import Status, atomic_json, now
from .ground_detection_cli import detection_ap, operating_metrics, THRESHOLDS


def profiles(root, suite="standard"):
    result = [DetectionProfile("current_yolo11s_960_fp32", str(root / "yolo11s.pt"), 960, fp16=False)]
    if suite == "advanced":
        for architecture in ("yolo11s", "yolo11m", "yolo26s", "yolo26m", "yolo11l", "yolo11x"):
            for size in ((1280, 1536) if architecture == "yolo11x" else (1280, 1536, 1920)):
                result.append(DetectionProfile(f"{architecture}_{size}_tta", str(root / f"{architecture}.pt"), size, augment=True))
        return result
    if suite != "standard":
        raise ValueError("Unknown frozen candidate suite")
    for architecture in ("yolo11s", "yolo11m", "yolo26s", "yolo26m"):
        for size in ((1280, 1536) if architecture == "yolo11s" else (960, 1280, 1536)):
            result.append(DetectionProfile(f"{architecture}_{size}_fp16", str(root / f"{architecture}.pt"), size))
    result += [DetectionProfile("yolo11s_1536_tta", str(root / "yolo11s.pt"), 1536, augment=True),
               DetectionProfile("yolo11s_tiles768", str(root / "yolo11s.pt"), 1536, tile_size=768),
               DetectionProfile("yolo26s_tiles768", str(root / "yolo26s.pt"), 1536, tile_size=768)]
    return result


def raw_counts(rows, threshold):
    errors = [sum(prediction[4] >= threshold for prediction in row["predictions"]) - len(row["truth"]) for row in rows]
    return {"mae": sum(abs(value) for value in errors) / len(errors),
            "rmse": math.sqrt(sum(value * value for value in errors) / len(errors)),
            "bias": sum(errors) / len(errors), "threshold": threshold,
            "definition": "Unfiltered person detections versus marked-person annotations, including predictions on ignored regions. This is not all-visible-human ground truth."}


def summarize(rows, threshold=.35):
    return {"detection": detection_ap(rows), "fixed": operating_metrics(rows, threshold),
            "raw_counts": raw_counts(rows, threshold)}


def evaluate(profile, manifest, directory, state, partition):
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    state.update(status="evaluating", activeStage=f"{partition}: {profile.name}",
                 evaluationImagesDone=0, evaluationImagesTotal=len(manifest))
    model = None
    rows, timings = [], []
    try:
        model = YOLO(profile.checkpoint)
        if model.names.get(0) != "person":
            raise ValueError("Official person class 0 is required")
        # Fresh model per profile; initialize YOLO26 non-e2e head BEFORE fusion.
        first_bytes = Path(manifest[0]["image"]).read_bytes()
        first = cv2.imdecode(np.frombuffer(first_bytes, np.uint8), cv2.IMREAD_COLOR)
        predict_boxes(model, first, profile)
        if bool(model.predictor.model.fp16) != profile.fp16:
            raise RuntimeError("Actual inference precision differs from the declared profile")
        torch.cuda.reset_peak_memory_stats()
        with (directory / f"{partition}_{profile.name}_predictions.jsonl").open("w", encoding="utf-8") as output:
            for index, item in enumerate(manifest):
                torch.cuda.synchronize()
                started = time.perf_counter()
                image_bytes = Path(item["image"]).read_bytes()
                if hashlib.sha256(image_bytes).hexdigest() != item["image_sha256"]:
                    raise ValueError("Source image changed since the frozen manifest")
                frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("Unreadable source image")
                predictions = predict_boxes(model, frame, profile)
                torch.cuda.synchronize()
                timings.append((time.perf_counter() - started) * 1000)
                rows.append({**item, "predictions": predictions})
                output.write(json.dumps({"sequence": item["sequence"], "frame": item["frame"],
                                         "timestamp_seconds": item["timestamp_seconds"], "boxes_scores": predictions}) + "\n")
                if index % 10 == 0 or index + 1 == len(manifest):
                    state.update(evaluationImagesDone=index + 1)
        result = {"name": profile.name, "profile": asdict(profile), "partition": partition,
                  "checkpoint_sha256": hashlib.sha256(Path(profile.checkpoint).read_bytes()).hexdigest(),
                  **summarize(rows), "speed": {"mean_decode_predict_ms": float(np.mean(timings)),
                  "p95_decode_predict_ms": float(np.percentile(timings, 95)),
                  "images_per_second": 1000 / float(np.mean(timings)),
                  "peak_allocated_vram_mb": torch.cuda.max_memory_allocated() / 2**20},
                  "per_scene": {name: summarize([row for row in rows if row["sequence"] == name]) for name in sorted({row["sequence"] for row in rows})}}
        if partition == "calibration":
            sweep = [operating_metrics(rows, value) for value in THRESHOLDS]
            guarded = [item for item in sweep if item["precision"] >= .8]
            selected = max(guarded, key=lambda item: (item["f1"], item["precision"]))["threshold"] if guarded else .35
            result.update(calibrated_threshold=selected, calibration_sweep=sweep, precision_guard_met=bool(guarded))
        atomic_json(directory / f"{partition}_{profile.name}.json", result)
        return result, rows
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def choose(calibrations):
    """Predeclared AP target, with count error and runtime as exact-score tie breakers."""
    return max(calibrations, key=lambda item: (item["detection"]["map50"], -item["fixed"]["count_mae"], item["speed"]["images_per_second"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--suite", choices=("standard", "advanced"), default="standard")
    args = parser.parse_args()
    if not args.name.replace("_", "").isalnum():
        parser.error("Unsafe experiment name")
    root = Path(__file__).resolve().parents[2]
    directory = root / "runs/experiments" / args.name
    directory.mkdir(exist_ok=False)
    state = Status(directory)
    inhibited = False
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        torch.set_num_threads(4)
        if os.name == "nt":
            inhibited = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        original = json.loads((root / "runs/experiments/ground_detection_20261013_corrected/manifest.json").read_text())
        calibration = [row for row in original["images"] if row["partition"] == "calibration"]
        report = [row for row in original["images"] if row["partition"] == "report"]
        candidates = profiles(root, args.suite)
        plan = {"createdAt": now(), "suite": args.suite, "target": .7, "targetMetric": "ground development mAP50", "profiles": [asdict(profile) for profile in candidates],
                "calibration_images": len(calibration), "report_images": len(report),
                "selection": "Highest calibration mAP50; exact ties by lower fixed-.35 adjusted count MAE then faster inference. Winner frozen before report prediction. No report-driven reselection.",
                "threshold_selection": "Calibration F1 maximum with precision >=.80 over .05:.05:.65, otherwise .35; fixed-.35 comparison also retained.",
                "scoring": original["protocol"]["matching"], "ap": original["protocol"]["ap"],
                "inference": "conf=.001, class0, per-pass and final max_det300, NMS IoU=.5. FP16 settings recorded per profile. YOLO26 non-e2e head selected before first inference. Tiles add full-image plus20%overlap tiles at960px; internal cut boxes rejected; global NMS.5.",
                "limitations": "MOT17 public training scenes already repeatedly inspected by this project; development audit only, not independent final test. No tracking, identities, embeddings or cross-camera matching.",
                "split_sha256": {split: hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest() for split, rows in (("calibration", calibration), ("report", report))}}
        atomic_json(directory / "plan.json", plan)
        atomic_json(directory / "manifest.json", {**original, "protocol": plan})
        state.update(architecture="Stronger pretrained ground detectors + resolution/tiling", target=.7, targetMetric=plan["targetMetric"],
                     datasetDescription="137 fixed calibration images select the profile; 78 fixed development report images score the frozen winner. No new training.",
                     dataset={"train": {"images": 0}, "val": {"images": len(calibration)}, "report": {"images": len(report)}},
                     profilesTotal=len(candidates), profilesDone=0, gpu=torch.cuda.get_device_name(0), idleSleepInhibited=inhibited)
        atomic_json(directory.parent / "active.json", {"name": args.name})
        calibrations, failures = [], []
        for index, profile in enumerate(candidates):
            try:
                measured, _ = evaluate(profile, calibration, directory, state, "calibration")
                calibrations.append(measured)
            except Exception as error:
                failure = {"profile": profile.name, "error": str(error), "traceback": traceback.format_exc()}
                failures.append(failure)
                atomic_json(directory / f"failed_{profile.name}.json", failure)
                if index == 0:
                    raise
            state.update(profilesDone=index + 1)
        winner = choose(calibrations)
        frozen = {"frozenAt": now(), "profile": winner["profile"], "checkpoint_sha256": winner["checkpoint_sha256"],
                  "calibration_map50": winner["detection"]["map50"], "operating_threshold": winner["calibrated_threshold"],
                  "selection_rule": plan["selection"], "report_scored_yet": False}
        atomic_json(directory / "selection.json", frozen)
        selected = DetectionProfile(**winner["profile"])
        measured_reports = []
        for profile in [candidates[0]] + ([] if selected.name == candidates[0].name else [selected]):
            measured, rows = evaluate(profile, report, directory, state, "report")
            selected_threshold = next(item["calibrated_threshold"] for item in calibrations if item["name"] == profile.name)
            measured["calibrated"] = {"threshold": selected_threshold, **summarize(rows, selected_threshold)}
            atomic_json(directory / f"report_{profile.name}.json", measured)
            measured_reports.append(measured)
        baseline = measured_reports[0]
        candidate = measured_reports[-1]
        reached = candidate["detection"]["map50"] > .7
        favourable = (candidate["detection"]["map50"] > baseline["detection"]["map50"] and
                      candidate["fixed"]["recall"] >= baseline["fixed"]["recall"] and
                      candidate["fixed"]["count_mae"] <= baseline["fixed"]["count_mae"] and
                      candidate["fixed"]["precision"] >= .8)
        comparison = {"finishedAt": now(), "target_reached": reached, "deployment_review_favourable": favourable,
                      "checkpoint_promoted": False, "selected_profile": asdict(selected), "calibrations": calibrations,
                      "reports": measured_reports, "errors": failures, "limitation": plan["limitations"],
                      "evaluations": [{"name": item["name"], "detection": {"metrics/mAP50(B)": item["detection"]["map50"], "metrics/mAP50-95(B)": item["detection"]["map5095"]},
                          "counts": {"selected_threshold": .35, "report": {"mae": item["fixed"]["count_mae"], "rmse": item["fixed"]["count_rmse"], "threshold": .35}}} for item in measured_reports]}
        atomic_json(directory / "comparison.json", comparison)
        lines = ["# Frozen ground-detection score target", "", plan["limitations"], "", f"Target strictly greater than70% mAP50: {reached}",
                 f"Selected from calibration only: {selected.name}", f"Deployment review favourable at fixed .35: {favourable}; no automatic promotion.", "",
                 "| Profile | Report mAP50 | mAP50–95 | Precision at .35 | Recall at .35 | Adjusted count MAE | Raw count MAE | FPS |",
                 "|---|---|---|---|---|---|---|---|"]
        for item in measured_reports:
            metric = item["fixed"]
            lines.append(f"| {item['name']} | {item['detection']['map50']:.2%} | {item['detection']['map5095']:.2%} | {metric['precision']:.2%} | {metric['recall']:.2%} | {metric['count_mae']:.2f} | {item['raw_counts']['mae']:.2f} | {item['speed']['images_per_second']:.1f} |")
        lines += ["", "Adjusted count error neutralizes detections on ignored annotations using ground truth. Raw error does not; neither is verified all-visible-human counting accuracy.", "", "## Calibration-only search", "", "| Profile | Calibration mAP50 | Count MAE at .35 |", "|---|---|---|"]
        for item in calibrations:
            lines.append(f"| {item['name']} | {item['detection']['map50']:.2%} | {item['fixed']['count_mae']:.2f} |")
        if failures:
            lines += ["", "## Failed profiles", *[f"- {item['profile']}: {item['error']}" for item in failures]]
        (directory / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        state.update(status="completed", finishedAt=now(), comparison=comparison, reportPath=str(directory / "REPORT.md"))
    except BaseException:
        state.update(status="failed", error=traceback.format_exc(), finishedAt=now())
        raise
    finally:
        if inhibited:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
