"""Record explicit anonymous QR tags only, with tag-only evidence and no person association."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .marker_codes import MarkerReader


class MarkerSession:
    def __init__(self, directory: Path, codes: list[str]):
        self.directory = Path(directory) / "marker_evidence"
        self.directory.mkdir()
        self.reader = MarkerReader(codes)
        self.log = (Path(directory) / "markers.jsonl").open("w", encoding="utf-8")
        self.frames = 0
        self.failures = 0
        self.observations = {}
        self.duplicates = set()
        self._last_frame = -1

    def read(self, frame, frame_index: int, seconds: float):
        import cv2
        import numpy as np

        if type(frame_index) is not int or frame_index <= self._last_frame:
            raise ValueError("Marker reads require distinct, increasing source frame indexes")
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Marker timestamps must be finite, nonnegative clip seconds")
        self._last_frame = frame_index
        self.frames += 1
        try:
            found = self.reader.read(frame)
            duplicated = set(self.reader.duplicate_codes)
        except (cv2.error, RuntimeError, ValueError):
            self.failures += 1
            self.log.write(json.dumps({"frame_index": frame_index, "timestamp_seconds": seconds,
                                       "read_failed": True, "markers": [], "duplicate_codes": []}) + "\n")
            return []
        self.duplicates.update(duplicated)
        seen = set()
        for marker in found:
            code = marker["code"]
            if code in seen:
                continue
            seen.add(code)
            observation = self.observations.setdefault(code, {
                "first_frame": frame_index, "last_frame": frame_index,
                "first_seconds": seconds, "last_seconds": seconds,
                "read_frames": 0, "evidence_file": None,
            })
            observation.update(last_frame=frame_index, last_seconds=seconds,
                               read_frames=observation["read_frames"] + 1)
            if observation["evidence_file"] is None and code not in duplicated:
                # Rectify precisely the decoded QR quadrilateral, never a face/person crop.
                quad = np.asarray(marker["corners"], dtype=np.float32)
                target = np.asarray([[0, 0], [255, 0], [255, 255], [0, 255]], dtype=np.float32)
                evidence = cv2.warpPerspective(frame, cv2.getPerspectiveTransform(quad, target), (256, 256))
                filename = f"{code}.jpg"
                if not cv2.imwrite(str(self.directory / filename), evidence):
                    raise RuntimeError("Could not preserve decoded QR tag evidence")
                observation["evidence_file"] = filename
        self.log.write(json.dumps({"frame_index": frame_index, "timestamp_seconds": seconds,
                                   "read_failed": False, "markers": found,
                                   "duplicate_codes": sorted(duplicated)}) + "\n")
        return found

    def summary(self, frames_processed: int):
        return {"frames_processed": frames_processed, "frames_scanned": self.frames,
                "read_failures": self.failures, "observations": self.observations,
                "duplicate_codes": sorted(self.duplicates)}

    def close(self):
        self.log.close()
