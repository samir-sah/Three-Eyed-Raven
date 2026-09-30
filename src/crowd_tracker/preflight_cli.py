"""Check that a video, RTSP, or HTTP camera source is readable before inference."""

from __future__ import annotations

import argparse
import json
from urllib.parse import urlsplit, urlunsplit

import cv2


def redact_source(source: str) -> str:
    """Hide URL credentials while retaining enough context for a preflight report."""
    parsed = urlsplit(source)
    if not parsed.scheme or not parsed.netloc:
        return source
    hostname = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    netloc = f"***@{hostname}{port}" if parsed.username else f"{hostname}{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify that a video or camera stream can be opened and sampled.")
    parser.add_argument("--source", required=True, help="Local video path, RTSP URL, or HTTP stream URL.")
    parser.add_argument("--sample-frames", type=int, default=3, help="Frames to read during preflight.")
    args = parser.parse_args()
    if args.sample_frames < 1:
        raise ValueError("sample-frames must be at least 1.")

    capture = cv2.VideoCapture(args.source)
    if not capture.isOpened():
        raise RuntimeError("Could not open the supplied source. Check network reachability, credentials, codec support, and firewall rules.")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        read = 0
        for _ in range(args.sample_frames):
            ok, _frame = capture.read()
            if not ok:
                break
            read += 1
    finally:
        capture.release()
    report = {
        "source": redact_source(args.source),
        "opened": True,
        "sample_frames_requested": args.sample_frames,
        "sample_frames_read": read,
        "readable": read == args.sample_frames,
        "resolution": [width, height],
        "source_fps": fps,
    }
    print(json.dumps(report, indent=2))
    if not report["readable"]:
        raise RuntimeError("The source opened but did not yield every requested frame.")


if __name__ == "__main__":
    main()
