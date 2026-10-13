"""Local two-view anonymous detection/counting, deliberately without identity logic."""
from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

from .extended_detection_cli import atomic_json, now
from .storage import RunStore
from .detection_profiles import DetectionProfile, predict_boxes


def count_boxes(boxes, width, height):
    """Clip valid image-level detections; preserve confidence, never attach an ID."""
    clean = []
    for values in boxes:
        x1, y1, x2, y2, score = map(float, values[:5])
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2, score)) or not 0 <= score <= 1:
            continue
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 > x1 and y2 > y1:
            clean.append([x1, y1, x2, y2, score])
    return clean


def validate_config(config):
    cameras = config.get("cameras", [])
    if len(cameras) != 2 or {camera.get("id") for camera in cameras} != {"aerial", "ground"}:
        raise ValueError("Exactly one aerial and one ground video are required")
    maximum = config.get("max_frames_per_camera", 600)
    if maximum is not None and (type(maximum) is not int or not 1 <= maximum <= 20000):
        raise ValueError("Invalid frame limit")
    for camera in cameras:
        if not Path(camera["source"]).is_file() or not Path(camera["model"]).is_file():
            raise ValueError(f"Source or checkpoint missing for {camera['id']}")
        if camera.get("detection_profile"):
            profile = DetectionProfile(**camera["detection_profile"])
            if Path(profile.checkpoint).resolve() != Path(camera["model"]).resolve() or profile.imgsz != camera.get("imgsz"):
                raise ValueError("Approved profile must match the camera checkpoint and resolution")
        elif camera.get("imgsz", 640) not in (640, 960):
            raise ValueError("Higher resolutions require an explicit measured anonymous detector profile")
    if not 0 < config.get("confidence_threshold", .35) < 1:
        raise ValueError("Invalid confidence threshold")
    if any(key.startswith("reid") or key == "tracker" for key in config):
        raise ValueError("Identity/tracking settings are not accepted in anonymous analysis")
    return config


class BrowserVideoWriter:
    """H.264/yuv420p with fast-start metadata, suitable for browser seeking."""
    def __init__(self, directory, width, height, fps):
        import imageio_ffmpeg
        scale = min(1, 1280 / max(width, height))
        self.size = (max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2))
        self.log = (directory / "encoding.log").open("wb")
        self.process = subprocess.Popen([
            imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "warning", "-n",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{self.size[0]}x{self.size[1]}",
            "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
            "-threads", "2", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(directory / "browser.mp4")], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=self.log, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def write(self, frame):
        import cv2
        self.process.stdin.write(cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA).tobytes())

    def close(self, failed=False):
        try:
            self.process.stdin.close()
            code = self.process.wait(timeout=60)
            if code and not failed:
                raise RuntimeError("Video encoding failed; inspect encoding.log")
        finally:
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=10)
            self.log.close()


def analyse_camera(camera, config, output, progress):
    import cv2
    import torch
    from ultralytics import YOLO
    directory = output / camera["id"]
    directory.mkdir()
    capture = cv2.VideoCapture(camera["source"])
    writer = None
    failed = True
    started = time.perf_counter()
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot decode {camera['id']} video")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video frame rate could not be read")
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width < 2 or height < 2:
            raise ValueError("Video dimensions could not be read")
        model = YOLO(camera["model"])
        if model.names.get(0) not in ("person", "pedestrian", "people"):
            raise ValueError("Checkpoint does not declare person class 0")
        writer = BrowserVideoWriter(directory, width, height, fps)
        device = config.get("device", "auto")
        if device == "auto":
            device = 0 if torch.cuda.is_available() else "cpu"
        profile = DetectionProfile(**camera["detection_profile"]) if camera.get("detection_profile") else None
        if profile and str(device) == "cpu":
            profile = replace(profile, fp16=False)
        count = peak = total_detections = 0
        with (directory / "counts.jsonl").open("w", encoding="utf-8") as counts:
            while config.get("max_frames_per_camera") is None or count < config["max_frames_per_camera"]:
                ok, frame = capture.read()
                if not ok:
                    break
                source_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
                threshold = config.get("confidence_threshold", .35)
                if profile:
                    raw = [box for box in predict_boxes(model, frame, profile, device=device) if box[4] >= threshold]
                else:
                    result = model.predict(frame, imgsz=camera.get("imgsz", 640), conf=threshold, iou=.5,
                                           max_det=1000, classes=[0], device=device, verbose=False, save=False)[0]
                    raw = result.boxes.data[:, :5].detach().cpu().tolist()
                boxes = count_boxes(raw, width, height)
                detected = len(boxes)
                peak = max(peak, detected)
                total_detections += detected
                # Every decoded frame is logged, including an empty scene.
                counts.write(json.dumps({"frame_index": count, "timestamp_seconds": count / fps,
                                         "source_timestamp_seconds": source_ms / 1000 if math.isfinite(source_ms) else None,
                                         "count": detected, "boxes_scores": boxes}) + "\n")
                annotated = cv2.resize(frame, writer.size, interpolation=cv2.INTER_AREA)
                sx, sy = writer.size[0] / width, writer.size[1] / height
                for x1, y1, x2, y2, _ in boxes:
                    cv2.rectangle(annotated, (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy)), (255, 255, 255), 2)
                cv2.rectangle(annotated, (8, 8), (220, 43), (18, 18, 18), -1)
                cv2.putText(annotated, f"People detected: {detected}", (17, 31), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
                writer.write(annotated)
                count += 1
                if count % 25 == 0:
                    atomic_json(progress, {"status": "processing", "camera": camera["id"], "frames": count, "updatedAt": now()})
        if not count:
            raise ValueError(f"No readable frames in {camera['id']} video")
        failed = False
        elapsed = time.perf_counter() - started
        return {"camera_id": camera["id"], "label": camera.get("label"), "source": camera["source"], "model": camera["model"],
                "imgsz": camera.get("imgsz", 640), "source_fps": fps, "source_resolution": [width, height],
                "frames_processed": count, "peak_count": peak, "mean_count": total_detections / count,
                "duration_seconds": count / fps, "processing_fps": count / elapsed, "alert_count": 0,
                "alerts_enabled": False, "output_resolution": list(writer.size),
                "runtime_detection_profile": asdict(profile) if profile else None,
                "runtime_device": str(device)}
    finally:
        capture.release()
        if writer:
            writer.close(failed)
        if "model" in locals():
            del model
        gc.collect()
        torch.cuda.empty_cache()


def run(config):
    config = validate_config(config)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    progress = output / "progress.json"
    started = time.perf_counter()
    try:
        cameras = [analyse_camera(camera, config, output, progress) for camera in config["cameras"]]
        elapsed = time.perf_counter() - started
        frames = sum(camera["frames_processed"] for camera in cameras)
        summary = {"run_type": "multi_stream", "analysis_type": "anonymous_detection",
                   "completed_at": now(), "config": config, "cameras": cameras,
                   "total_frames_processed": frames, "elapsed_seconds": elapsed,
                   "aggregate_processing_fps": frames / elapsed, "alert_count": 0,
                   "limitations": "Per-frame detector estimates, not unique people or verified accuracy. Views are independent; their counts must not be added to infer a shared crowd. Playback uses each clip's nominal FPS, not camera clock synchronization. No crowd alert limit was configured."}
        atomic_json(output / "summary.json", summary)
        RunStore.for_output_dir(output).upsert(output.name, output, summary)
        atomic_json(progress, {"status": "completed", "updatedAt": now()})
        return summary
    except BaseException as error:
        atomic_json(progress, {"status": "failed", "error": str(error), "updatedAt": now()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
