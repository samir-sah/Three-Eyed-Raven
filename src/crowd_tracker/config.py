"""Configuration loading and validation for a single video pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ZoneConfig:
    id: str
    points: list[list[int]]
    occupancy_alert_threshold: int

    @classmethod
    def from_dict(cls, raw: dict) -> "ZoneConfig":
        points = raw.get("points", [])
        if not isinstance(raw.get("id"), str) or not raw["id"].strip():
            raise ValueError("Each zone needs a non-empty string id.")
        if len(points) < 3 or any(not isinstance(point, list) or len(point) != 2 for point in points):
            raise ValueError(f"Zone '{raw['id']}' must contain at least three [x, y] points.")
        threshold = raw.get("occupancy_alert_threshold")
        if not isinstance(threshold, int) or threshold < 1:
            raise ValueError(f"Zone '{raw['id']}' needs an integer occupancy_alert_threshold >= 1.")
        return cls(id=raw["id"], points=points, occupancy_alert_threshold=threshold)


@dataclass(frozen=True)
class AppConfig:
    source: str
    output_dir: str
    model: str = "yolo11n.pt"
    device: str = "auto"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.5
    frame_stride: int = 1
    max_frames: int | None = None
    display: bool = False
    save_video: bool = True
    tracker: str = "bytetrack.yaml"
    zones: list[ZoneConfig] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: str | Path) -> "AppConfig":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as file:
            raw = json.load(file)
        config = cls(
            source=raw["source"],
            output_dir=raw["output_dir"],
            model=raw.get("model", "yolo11n.pt"),
            device=raw.get("device", "auto"),
            confidence_threshold=raw.get("confidence_threshold", 0.35),
            iou_threshold=raw.get("iou_threshold", 0.5),
            frame_stride=raw.get("frame_stride", 1),
            max_frames=raw.get("max_frames"),
            display=raw.get("display", False),
            save_video=raw.get("save_video", True),
            tracker=raw.get("tracker", "bytetrack.yaml"),
            zones=[ZoneConfig.from_dict(zone) for zone in raw.get("zones", [])],
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.source:
            raise ValueError("source is required.")
        if not 0 < self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be in (0, 1].")
        if not 0 < self.iou_threshold <= 1:
            raise ValueError("iou_threshold must be in (0, 1].")
        if self.frame_stride < 1:
            raise ValueError("frame_stride must be >= 1.")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("max_frames must be null or >= 1.")
        zone_ids = [zone.id for zone in self.zones]
        if len(zone_ids) != len(set(zone_ids)):
            raise ValueError("Zone ids must be unique.")

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CameraStreamConfig:
    id: str
    source: str
    zones: list[ZoneConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict) -> "CameraStreamConfig":
        identifier = raw.get("id")
        source = raw.get("source")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("Each camera requires a non-empty string id.")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"Camera '{identifier}' requires a source.")
        return cls(
            id=identifier,
            source=source,
            zones=[ZoneConfig.from_dict(zone) for zone in raw.get("zones", [])],
        )


@dataclass(frozen=True)
class MultiStreamConfig:
    output_dir: str
    cameras: list[CameraStreamConfig]
    model: str = "yolo11n.pt"
    device: str = "auto"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.5
    frame_stride: int = 1
    max_frames_per_camera: int | None = None
    save_video: bool = True
    tracker: str = "bytetrack.yaml"

    @classmethod
    def from_file(cls, path: str | Path) -> "MultiStreamConfig":
        with Path(path).open("r", encoding="utf-8") as file:
            raw = json.load(file)
        config = cls(
            output_dir=raw["output_dir"],
            cameras=[CameraStreamConfig.from_dict(camera) for camera in raw.get("cameras", [])],
            model=raw.get("model", "yolo11n.pt"),
            device=raw.get("device", "auto"),
            confidence_threshold=raw.get("confidence_threshold", 0.35),
            iou_threshold=raw.get("iou_threshold", 0.5),
            frame_stride=raw.get("frame_stride", 1),
            max_frames_per_camera=raw.get("max_frames_per_camera"),
            save_video=raw.get("save_video", True),
            tracker=raw.get("tracker", "bytetrack.yaml"),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.output_dir:
            raise ValueError("output_dir is required.")
        if len(self.cameras) < 2:
            raise ValueError("Multi-stream runs require at least two cameras.")
        camera_ids = [camera.id for camera in self.cameras]
        if len(camera_ids) != len(set(camera_ids)):
            raise ValueError("Camera ids must be unique.")
        if not 0 < self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be in (0, 1].")
        if not 0 < self.iou_threshold <= 1:
            raise ValueError("iou_threshold must be in (0, 1].")
        if self.frame_stride < 1:
            raise ValueError("frame_stride must be >= 1.")
        if self.max_frames_per_camera is not None and self.max_frames_per_camera < 1:
            raise ValueError("max_frames_per_camera must be null or >= 1.")

    def as_dict(self) -> dict:
        return asdict(self)
