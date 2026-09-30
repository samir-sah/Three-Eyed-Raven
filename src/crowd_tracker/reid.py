"""Conservative, explainable cross-camera appearance matching baseline.

This module deliberately produces review candidates, not identity decisions. Its HSV
appearance embedding is a lightweight baseline suitable for verifying pipeline flow;
it is not a replacement for a trained aerial-ground person Re-ID model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .models import TrackObservation


@dataclass
class _GalleryEntry:
    camera_id: str
    track_id: int
    first_timestamp: float
    last_timestamp: float
    samples: list[tuple[float, np.ndarray]] = field(default_factory=list)
    best_quality: float = 0.0
    best_crop: np.ndarray | None = None


@dataclass(frozen=True)
class TrackDescriptor:
    camera_id: str
    track_id: int
    first_timestamp: float
    last_timestamp: float
    sample_count: int
    embedding: np.ndarray
    thumbnail_path: str | None = None


class AppearanceGallery:
    """Maintains a small set of the best crops for each local track."""

    def __init__(self, min_box_size: int = 32, max_samples_per_track: int = 6) -> None:
        self._min_box_size = min_box_size
        self._max_samples = max_samples_per_track
        self._entries: dict[tuple[str, int], _GalleryEntry] = {}

    def add(self, frame: np.ndarray, observation: TrackObservation) -> None:
        crop = self._crop(frame, observation.bbox_xyxy)
        if crop is None:
            return
        quality = self._quality(crop, observation.confidence)
        if quality <= 0:
            return
        key = (observation.camera_id, observation.track_id)
        entry = self._entries.get(key)
        if entry is None:
            entry = _GalleryEntry(
                camera_id=observation.camera_id,
                track_id=observation.track_id,
                first_timestamp=observation.timestamp_seconds,
                last_timestamp=observation.timestamp_seconds,
            )
            self._entries[key] = entry
        entry.first_timestamp = min(entry.first_timestamp, observation.timestamp_seconds)
        entry.last_timestamp = max(entry.last_timestamp, observation.timestamp_seconds)
        entry.samples.append((quality, self._embedding(crop)))
        entry.samples.sort(key=lambda sample: sample[0], reverse=True)
        del entry.samples[self._max_samples :]
        if quality > entry.best_quality:
            entry.best_quality = quality
            entry.best_crop = crop.copy()

    def descriptors(self, thumbnail_paths: dict[tuple[str, int], str] | None = None) -> list[TrackDescriptor]:
        result: list[TrackDescriptor] = []
        for entry in self._entries.values():
            if not entry.samples:
                continue
            qualities = np.asarray([quality for quality, _ in entry.samples], dtype=np.float32)
            vectors = np.stack([vector for _, vector in entry.samples])
            averaged = np.average(vectors, axis=0, weights=qualities)
            norm = float(np.linalg.norm(averaged))
            if norm == 0:
                continue
            result.append(TrackDescriptor(
                camera_id=entry.camera_id,
                track_id=entry.track_id,
                first_timestamp=entry.first_timestamp,
                last_timestamp=entry.last_timestamp,
                sample_count=len(entry.samples),
                embedding=averaged / norm,
                thumbnail_path=(thumbnail_paths or {}).get((entry.camera_id, entry.track_id)),
            ))
        return result

    def write_thumbnails(self, output_directory: str) -> dict[tuple[str, int], str]:
        """Save the strongest crop per track as compact review evidence."""
        from pathlib import Path

        import cv2

        directory = Path(output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths: dict[tuple[str, int], str] = {}
        for key, entry in self._entries.items():
            if entry.best_crop is None:
                continue
            height, width = entry.best_crop.shape[:2]
            target_width = min(180, width)
            target_height = max(1, round(height * target_width / width))
            image = cv2.resize(entry.best_crop, (target_width, target_height), interpolation=cv2.INTER_AREA)
            name = f"{entry.camera_id}_track_{entry.track_id}.jpg"
            destination = directory / name
            if cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 88]):
                paths[key] = f"reid_crops/{name}"
        return paths

    def _crop(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, x2 = max(0, x1), min(frame_width, x2)
        y1, y2 = max(0, y1), min(frame_height, y2)
        if x2 - x1 < self._min_box_size or y2 - y1 < self._min_box_size:
            return None
        return frame[y1:y2, x1:x2]

    def _quality(self, crop: np.ndarray, confidence: float) -> float:
        height, width = crop.shape[:2]
        size_score = min(1.0, min(width, height) / (self._min_box_size * 2))
        return max(0.0, min(1.0, confidence)) * size_score

    @staticmethod
    def _embedding(crop: np.ndarray) -> np.ndarray:
        """HSV colour distribution baseline, L2-normalized for cosine similarity."""
        import cv2

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
        histogram = histogram.astype(np.float32)
        norm = float(np.linalg.norm(histogram))
        if norm == 0:
            return histogram
        return histogram / norm


class CrossCameraMatcher:
    def __init__(self, candidate_threshold: float = 0.72, top_k: int = 3) -> None:
        self._threshold = candidate_threshold
        self._top_k = top_k

    def rank(self, descriptors: list[TrackDescriptor]) -> list[dict]:
        candidates: list[dict] = []
        for probe in descriptors:
            ranked: list[tuple[float, TrackDescriptor]] = []
            for gallery in descriptors:
                if probe.camera_id == gallery.camera_id:
                    continue
                similarity = float(np.clip(np.dot(probe.embedding, gallery.embedding), -1.0, 1.0))
                ranked.append((similarity, gallery))
            ranked.sort(key=lambda item: item[0], reverse=True)
            for rank, (similarity, gallery) in enumerate(ranked[: self._top_k], start=1):
                candidates.append({
                    "probe_camera_id": probe.camera_id,
                    "probe_track_id": probe.track_id,
                    "gallery_camera_id": gallery.camera_id,
                    "gallery_track_id": gallery.track_id,
                    "rank": rank,
                    "similarity": round(similarity, 4),
                    "status": "review" if similarity >= self._threshold else "below_threshold",
                    "probe_samples": probe.sample_count,
                    "gallery_samples": gallery.sample_count,
                    "probe_thumbnail": probe.thumbnail_path,
                    "gallery_thumbnail": gallery.thumbnail_path,
                })
        return candidates
