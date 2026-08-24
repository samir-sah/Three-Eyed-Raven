"""Ultralytics YOLO + ByteTrack adapter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrackedPerson:
    track_id: int
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float


class ByteTrackPersonTracker:
    """Returns local track IDs. It has no cross-camera identity meaning."""

    def __init__(self, model_path: str, device: str, confidence: float, iou: float, tracker: str) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("Missing dependency. Run: pip install -r requirements.txt") from error
        self._model = YOLO(model_path)
        self._device = None if device == "auto" else device
        self._confidence = confidence
        self._iou = iou
        self._tracker = tracker

    def track(self, frame: np.ndarray) -> list[TrackedPerson]:
        results = self._model.track(
            frame,
            persist=True,
            classes=[0],
            conf=self._confidence,
            iou=self._iou,
            tracker=self._tracker,
            device=self._device,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        ids = boxes.id.int().cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        return [
            TrackedPerson(
                track_id=int(track_id),
                bbox_xyxy=tuple(map(int, box)),
                confidence=float(score),
            )
            for box, track_id, score in zip(xyxy, ids, confidences, strict=True)
        ]
