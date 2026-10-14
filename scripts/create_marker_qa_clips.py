"""Create clearly synthetic tag-only clips for local QR demo verification, not a person benchmark."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from crowd_tracker.anonymous_cli import BrowserVideoWriter
from crowd_tracker.marker_codes import MarkerReader, generate_qr_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    roles = {"aerial": ["V07", "V08"], "ground": ["V07", "V09"]}
    frame_count, fps = 16, 2
    for role, codes in roles.items():
        directory = root / role
        directory.mkdir()
        writer = BrowserVideoWriter(directory, 960, 540, fps)
        try:
            cards = [generate_qr_image(code, 300) for code in codes]
            for index in range(frame_count):
                frame = np.full((540, 960, 3), 238, dtype=np.uint8)
                cv2.putText(frame, f"SYNTHETIC QR QA - {role.upper()} SLOT", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 0, 0), 2)
                cv2.putText(frame, "No people. Tests visible codes only.", (50, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1)
                for position, card in enumerate(cards):
                    x, y = 110 + position * 420, 140 + index % 3
                    frame[y:y + 300, x:x + 300] = card
                writer.write(frame)
        finally:
            writer.close()
        capture = cv2.VideoCapture(str(directory / "browser.mp4"))
        reader = MarkerReader(["V07", "V08", "V09"])
        observed = {code: 0 for code in codes}
        decoded = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                decoded += 1
                for code in {marker["code"] for marker in reader.read(frame)}:
                    if code in observed:
                        observed[code] += 1
        finally:
            capture.release()
        if decoded != frame_count or any(reads < 3 for reads in observed.values()):
            raise RuntimeError(f"Synthetic clip failed its own readability check: {role}, {observed}")
    (root / "FIXTURE.json").write_text(json.dumps({
        "kind": "synthetic_tag_only_qa", "frames_per_camera": frame_count, "fps": fps,
        "registered_codes": ["V07", "V08", "V09"], "expected_camera_codes": roles,
        "expected_shared_codes": ["V07"], "not_a_person_detection_benchmark": True,
    }, indent=2), encoding="utf-8")
    print(f"Synthetic QA clips created in {root}; expected shared code V07, not a person identity.")


if __name__ == "__main__":
    main()
