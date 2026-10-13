"""Isolated anonymous detector profiles; never tracking or identity matching.

Each call returns only original-image coordinates and a detection score. A fresh
model must be constructed by the caller when changing profiles: Ultralytics
initializes inference precision and end-to-end mode on its first prediction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DetectionProfile:
    name: str
    checkpoint: str
    imgsz: int
    tile_size: int = 0
    augment: bool = False
    fp16: bool = True
    end2end: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", self.name):
            raise ValueError("Profile name must be a safe alphanumeric, underscore or hyphen identifier")
        if not isinstance(self.checkpoint, str) or not self.checkpoint.strip():
            raise ValueError("A nonempty checkpoint path is required")
        if type(self.imgsz) is not int or not 320 <= self.imgsz <= 1920 or self.imgsz % 32:
            raise ValueError("imgsz must be a multiple of 32 between 320 and 1920")
        if type(self.tile_size) is not int or (
            self.tile_size != 0 and (not 320 <= self.tile_size <= 4096 or self.tile_size % 32)
        ):
            raise ValueError("tile_size must be zero or a multiple of 32 between 320 and 4096")
        if any(type(value) is not bool for value in (self.augment, self.fp16, self.end2end)):
            raise ValueError("augment, fp16 and end2end must be booleans")


def predict_boxes(model, frame, profile: DetectionProfile, *, device=0) -> list[list[float]]:
    """Predict anonymous ``[x1, y1, x2, y2, score]`` rows on one BGR image.

    Tiled profiles combine a whole-image pass with 20%-overlapping crops. Internal
    crop borders discard cut boxes, and global NMS removes repeated detections.
    Both tiled and whole-image profiles use the same final 300-detection cap.
    Inference runs sequentially; the image and merged boxes reside on CPU.
    Callers selecting CPU inference must explicitly set ``profile.fp16=False``.
    """
    import numpy as np
    import torch

    if not isinstance(profile, DetectionProfile):
        raise TypeError("profile must be a DetectionProfile")
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a nonempty H x W x 3 BGR image")
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("frame must be a nonempty H x W x 3 BGR image")

    whole = (0, 0, width, height)
    windows = [whole]
    if profile.tile_size:
        from .tiled_detection_cli import tile_windows

        windows.extend(window for window in tile_windows(width, height, profile.tile_size) if window != whole)

    predictions = []
    for window_index, (x1, y1, x2, y2) in enumerate(windows):
        result = model.predict(
            frame[y1:y2, x1:x2],
            imgsz=profile.imgsz if window_index == 0 else 960,
            conf=0.001,
            iou=0.5,
            max_det=300,
            classes=[0],
            device=device,
            verbose=False,
            save=False,
            augment=profile.augment,
            quantize=16 if profile.fp16 else 32,
            end2end=profile.end2end,
        )[0]
        if result.boxes is None:
            continue
        data = result.boxes.data.detach().to(device="cpu", dtype=torch.float32)
        if data.ndim != 2 or data.shape[1] != 6:
            # Seven-column tracking results contain identities and do not belong
            # in this helper. Fail explicitly rather than interpreting an ID as
            # confidence or accidentally persisting it.
            raise ValueError("Expected untracked detector boxes with six columns")
        boxes = data[:, :5].clone()
        keep = torch.isfinite(data).all(dim=1) & (data[:, 5] == 0)
        keep &= (boxes[:, 4] >= 0) & (boxes[:, 4] <= 1)
        if window_index:
            if x1:
                keep &= boxes[:, 0] > 2
            if y1:
                keep &= boxes[:, 1] > 2
            if x2 < width:
                keep &= boxes[:, 2] < x2 - x1 - 2
            if y2 < height:
                keep &= boxes[:, 3] < y2 - y1 - 2
        boxes = boxes[keep]
        boxes[:, [0, 2]] += x1
        boxes[:, [1, 3]] += y1
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, height)
        boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
        predictions.append(boxes)

    if not predictions:
        return []
    merged = torch.cat(predictions, dim=0)
    if not len(merged):
        return []
    if profile.tile_size:
        from torchvision.ops import nms

        indices = nms(merged[:, :4], merged[:, 4], 0.5)
    else:
        indices = torch.argsort(merged[:, 4], descending=True, stable=True)
    return merged[indices[:300]].tolist()
