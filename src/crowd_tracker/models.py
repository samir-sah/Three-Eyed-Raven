"""Domain events passed between pipeline modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TrackObservation:
    camera_id: str
    frame_index: int
    timestamp_seconds: float
    track_id: int
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    zones: tuple[str, ...]

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["bbox_xyxy"] = list(self.bbox_xyxy)
        payload["zones"] = list(self.zones)
        return payload


@dataclass(frozen=True)
class ZoneAlert:
    camera_id: str
    frame_index: int
    timestamp_seconds: float
    zone_id: str
    count: int
    threshold: int

    def as_dict(self) -> dict:
        return asdict(self)
