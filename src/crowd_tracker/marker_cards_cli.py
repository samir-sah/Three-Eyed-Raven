"""Generate anonymous printable QR cards, with no names or external payloads."""
import argparse
from pathlib import Path

from .marker_codes import generate_qr_image, normalize_codes


def main():
    import cv2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--codes", default=",".join(f"V{index:02d}" for index in range(1, 100)))
    args = parser.parse_args()
    codes = normalize_codes(args.codes.split(","))
    if not codes:
        parser.error("At least one code is required")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    for code in codes:
        if not cv2.imwrite(str(output / f"{code}.png"), generate_qr_image(code)):
            raise RuntimeError(f"Could not save {code}")
    print(f"Created {len(codes)} anonymous QR cards in {output}")


if __name__ == "__main__":
    main()
