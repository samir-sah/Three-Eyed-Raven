"""Opt-in volunteer QR markers, not person recognition or appearance matching.

Only pre-authorized anonymous ``TER:V01`` through ``TER:V99`` payloads are
returned. Other QR contents are discarded without being stored or logged.
"""
from __future__ import annotations

import math
import re


_CODE = re.compile(r"V(?:0[1-9]|[1-9][0-9])\Z")
_PAYLOAD = re.compile(r"TER:(V(?:0[1-9]|[1-9][0-9]))\Z")
_CONFIG_FIELDS = {"enabled", "confirmed_volunteer_demo", "allowed_codes", "min_reads"}


def normalize_codes(codes: list[str]) -> list[str]:
    """Validate strict anonymous code names and return sorted, distinct codes."""
    if not isinstance(codes, list) or any(not isinstance(code, str) or not _CODE.fullmatch(code) for code in codes):
        raise ValueError("Marker codes must be a list of V01 through V99 strings")
    return sorted(set(codes))


def _minimum_reads(value: int) -> int:
    if type(value) is not int or not 1 <= value <= 20000:
        raise ValueError("min_reads must be an integer between 1 and 20,000")
    return value


def validate_marker_config(config: dict | None) -> dict:
    """Marker reading is off unless explicitly enabled and volunteer-confirmed."""
    if config is None:
        config = {}
    if not isinstance(config, dict) or set(config) - _CONFIG_FIELDS:
        raise ValueError("Only enabled, confirmed_volunteer_demo, allowed_codes and min_reads are accepted")
    enabled = config.get("enabled", False)
    confirmed = config.get("confirmed_volunteer_demo", False)
    if type(enabled) is not bool or type(confirmed) is not bool:
        raise ValueError("Marker enablement and volunteer confirmation must be booleans")
    allowed = normalize_codes(config.get("allowed_codes", []))
    minimum = _minimum_reads(config.get("min_reads", 3))
    if enabled and (not confirmed or not allowed):
        raise ValueError("Enabled markers require explicit volunteer-demo confirmation and allowed codes")
    return {"enabled": enabled, "confirmed_volunteer_demo": confirmed, "allowed_codes": allowed, "min_reads": minimum}


def _same_geometry(first, second) -> bool:
    """Deduplicate decoder jitter, including shifted/reversed corner ordering."""
    import numpy as np

    for candidate in (second, second[::-1]):
        for shift in range(4):
            if np.max(np.linalg.norm(first - np.roll(candidate, shift, axis=0), axis=1)) <= 2:
                return True
    return False


class MarkerReader:
    def __init__(self, allowed_codes: list[str]):
        import cv2

        self.allowed_codes = frozenset(normalize_codes(allowed_codes))
        self.duplicate_codes: list[str] = []
        self._detector = cv2.QRCodeDetector()

    def read(self, frame) -> list[dict]:
        """Read authorized QR tags on a full frame, with original-image corners.

        ``duplicate_codes`` describes the latest frame only. Two separated tags
        with the same code remain separate observations and are ambiguous; they
        are never silently interpreted as one volunteer or one person.
        """
        import cv2
        import numpy as np

        self.duplicate_codes = []
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim not in (2, 3):
            raise ValueError("Marker decoding requires a uint8 grayscale or BGR frame")
        if not frame.shape[0] or not frame.shape[1] or (frame.ndim == 3 and frame.shape[2] != 3):
            raise ValueError("Marker decoding requires a nonempty grayscale or BGR frame")
        if not self.allowed_codes:
            return []
        try:
            ok, payloads, points, _ = self._detector.detectAndDecodeMulti(frame)
        except cv2.error:
            ok, payloads, points = False, (), None
        if not ok or not payloads or points is None or not any(payloads):
            try:
                payload, corners, _ = self._detector.detectAndDecode(frame)
            except cv2.error:
                raise RuntimeError("QR decoding failed for this frame") from None
            payloads = (payload,)
            points = None if corners is None else np.asarray(corners).reshape(1, 4, 2)
        if points is None:
            return []
        height, width = frame.shape[:2]
        clean = []
        geometries: dict[str, list] = {}
        for payload, corners in zip(payloads, points):
            match = _PAYLOAD.fullmatch(payload) if isinstance(payload, str) else None
            if not match or match.group(1) not in self.allowed_codes:
                continue
            code = match.group(1)
            coordinates = np.asarray(corners, dtype=np.float32)
            if coordinates.shape != (4, 2) or not np.isfinite(coordinates).all():
                continue
            if (coordinates[:, 0] < 0).any() or (coordinates[:, 0] > width - 1).any() or (coordinates[:, 1] < 0).any() or (coordinates[:, 1] > height - 1).any():
                continue
            if abs(cv2.contourArea(coordinates)) < 1:
                continue
            existing = geometries.setdefault(code, [])
            if any(_same_geometry(coordinates, previous) for previous in existing):
                continue
            existing.append(coordinates)
            clean.append({"code": code, "corners": coordinates.tolist()})
        self.duplicate_codes = sorted(code for code, locations in geometries.items() if len(locations) > 1)
        return sorted(clean, key=lambda item: (item["code"], min(corner[0] for corner in item["corners"])))


def _nonnegative_integer(value, default=0):
    return value if type(value) is int and value >= 0 else default


def _camera_markers(value: dict, minimum: int) -> dict:
    """Copy only safe aggregate/code evidence fields; discard unknown metadata."""
    if not isinstance(value, dict):
        value = {}
    processed = _nonnegative_integer(value.get("frames_processed"))
    scanned = min(processed, _nonnegative_integer(value.get("frames_scanned")))
    failures = min(scanned, _nonnegative_integer(value.get("read_failures")))
    duplicates = sorted({code for code in value.get("duplicate_codes", []) if isinstance(code, str) and _CODE.fullmatch(code)}) if isinstance(value.get("duplicate_codes", []), list) else []
    observations = {}
    raw = value.get("observations", {})
    for code, observation in (raw.items() if isinstance(raw, dict) else []):
        if not isinstance(code, str) or not _CODE.fullmatch(code) or not isinstance(observation, dict):
            continue
        first, last, reads = (observation.get(key) for key in ("first_frame", "last_frame", "read_frames"))
        first_seconds, last_seconds = (observation.get(key) for key in ("first_seconds", "last_seconds"))
        if any(type(number) is not int for number in (first, last, reads)) or not 0 <= first <= last < processed or not 1 <= reads <= min(scanned - failures, last - first + 1):
            continue
        if any(type(number) not in (int, float) or not math.isfinite(number) for number in (first_seconds, last_seconds)) or not 0 <= first_seconds <= last_seconds:
            continue
        safe = {"first_frame": first, "last_frame": last, "first_seconds": float(first_seconds),
                "last_seconds": float(last_seconds), "read_frames": reads,
                "evidence_file": f"{code}.jpg" if observation.get("evidence_file") == f"{code}.jpg" else None}
        observations[code] = safe
    observations = dict(sorted(observations.items()))
    return {"frames_processed": processed, "frames_scanned": scanned, "read_failures": failures,
            "observations": observations, "duplicate_codes": duplicates}


def summarize_marker_cameras(camera_summaries: list[dict], min_reads: int = 3) -> dict:
    """Summarize repeated marker codes across the two explicitly labeled views."""
    minimum = _minimum_reads(min_reads)
    if not isinstance(camera_summaries, (list, tuple)):
        raise ValueError("Camera summaries must be a list")
    cameras = {}
    for summary in camera_summaries:
        if not isinstance(summary, dict) or summary.get("camera_id") not in ("aerial", "ground"):
            continue
        camera = summary["camera_id"]
        if camera in cameras:
            raise ValueError("Duplicate camera summaries cannot be combined")
        cameras[camera] = _camera_markers(summary.get("markers", {}), minimum)
    reliable = {camera: {code for code, observation in markers["observations"].items()
                         if observation["read_frames"] >= minimum and code not in markers["duplicate_codes"]}
                for camera, markers in cameras.items()}
    shared = sorted(reliable.get("aerial", set()) & reliable.get("ground", set()))
    warning = "Shared codes are volunteer QR markers, not verified person identities. This controlled demonstration does not measure natural appearance matching. Unread, blurred or hidden tags can be missed."
    if any(camera["duplicate_codes"] for camera in cameras.values()):
        warning += " Codes seen at multiple locations are ambiguous and excluded from reliable shared counts."
    if len(cameras) != 2:
        warning += " Both camera summaries are required to report shared markers."
    return {"method": "explicit_qr_codes", "scope": "controlled_marker_demo", "min_reads": minimum,
            "cameras": cameras, "shared_codes": shared, "shared_count": len(shared), "warning": warning}


def generate_qr_image(code: str, size: int = 800):
    """Create a printable BGR QR card with a four-module quiet zone and caption."""
    import cv2
    import numpy as np

    normalize_codes([code])
    if type(size) is not int or not 128 <= size <= 4096:
        raise ValueError("Printed marker size must be an integer from 128 to 4096 pixels")
    if not hasattr(cv2, "QRCodeEncoder_create"):
        raise RuntimeError("This OpenCV installation does not include a QR encoder")
    parameters = cv2.QRCodeEncoder_Params()
    parameters.correction_level = cv2.QRCodeEncoder_CORRECT_LEVEL_H
    encoded = cv2.QRCodeEncoder_create(parameters).encode(f"TER:{code}")
    black_y, black_x = np.where(encoded == 0)
    modules = encoded[black_y.min():black_y.max() + 1, black_x.min():black_x.max() + 1]
    modules = cv2.copyMakeBorder(modules, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
    caption_height = max(26, round(size * .12))
    scale = max(1, (size - caption_height) // modules.shape[0])
    qr = cv2.resize(modules, (modules.shape[1] * scale, modules.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    card = np.full((size, size, 3), 255, dtype=np.uint8)
    x = (size - qr.shape[1]) // 2
    y = (size - caption_height - qr.shape[0]) // 2
    card[y:y + qr.shape[0], x:x + qr.shape[1]] = cv2.cvtColor(qr, cv2.COLOR_GRAY2BGR)
    font_scale, thickness = max(.45, size / 700), max(1, round(size / 400))
    text_width = cv2.getTextSize(code, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]
    cv2.putText(card, code, ((size - text_width) // 2, size - caption_height // 3), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return card
