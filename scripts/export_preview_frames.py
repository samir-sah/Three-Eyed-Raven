"""Create a browser-safe, controllable preview sequence from an annotated run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="live_preview", help="Artifact run name under artifacts/")
    parser.add_argument("--camera", help="Optional camera id for a multi-camera run")
    parser.add_argument("--stride", type=int, default=5, help="Keep one preview image every N source frames")
    parser.add_argument("--width", type=int, default=540, help="Preview image width in pixels")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_directory = project_root / "artifacts" / args.run
    if args.camera:
        if Path(args.camera).name != args.camera:
            raise ValueError("camera must be a simple camera id, not a path")
        run_directory /= args.camera
    source = run_directory / "annotated.mp4"
    output = run_directory / "preview_frames"
    if not source.exists():
        raise FileNotFoundError(f"Annotated video does not exist: {source}")
    if args.stride < 1 or args.width < 64:
        raise ValueError("stride must be >= 1 and width must be >= 64")

    output.mkdir(exist_ok=True)
    for old_frame in output.glob("frame_*.jpg"):
        old_frame.unlink()

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not read: {source}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 0
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_number = 0
    written = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_number % args.stride == 0:
                height, width = frame.shape[:2]
                resized_height = round(height * args.width / width)
                preview = cv2.resize(frame, (args.width, resized_height), interpolation=cv2.INTER_AREA)
                destination = output / f"frame_{written:03d}.jpg"
                if not cv2.imwrite(str(destination), preview, [cv2.IMWRITE_JPEG_QUALITY, 84]):
                    raise RuntimeError(f"Could not write: {destination}")
                written += 1
            frame_number += 1
    finally:
        capture.release()

    (output / "metadata.json").write_text(
        json.dumps(
            {
                "source_fps": source_fps,
                "source_frames": frame_number,
                "source_resolution": [source_width, source_height],
                "preview_stride": args.stride,
                "preview_frames": written,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {written} browser-preview frames to {output}")


if __name__ == "__main__":
    main()
