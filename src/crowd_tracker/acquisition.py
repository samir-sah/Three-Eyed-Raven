"""Bounded video acquisition primitives for file, RTSP, and HTTP sources."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class FramePacket:
    camera_id: str
    frame_index: int
    timestamp_seconds: float
    frame: np.ndarray


def preprocess_frame(frame: np.ndarray, size: tuple[int, int] | None = None, denoise: bool = False) -> np.ndarray:
    """Resize and optionally lightly denoise a BGR frame before downstream inference."""
    result = cv2.resize(frame, size, interpolation=cv2.INTER_AREA) if size else frame
    return cv2.fastNlMeansDenoisingColored(result, None, 3, 3, 7, 21) if denoise else result


class BoundedCapture:
    """Background capture with a bounded queue; oldest frames are dropped under load."""

    def __init__(
        self,
        camera_id: str,
        source: str,
        buffer_size: int = 8,
        preprocess: Callable[[np.ndarray], np.ndarray] | None = None,
        capture_factory: Callable[[str], object] = cv2.VideoCapture,
    ) -> None:
        if buffer_size < 1:
            raise ValueError("buffer_size must be >= 1.")
        self.camera_id, self.source = camera_id, source
        self._queue: queue.Queue[FramePacket] = queue.Queue(maxsize=buffer_size)
        self._preprocess = preprocess or (lambda frame: frame)
        self._capture_factory = capture_factory
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None

    def start(self) -> "BoundedCapture":
        if self._thread is not None:
            raise RuntimeError("Capture already started.")
        self._thread = threading.Thread(target=self._run, name=f"capture-{self.camera_id}", daemon=True)
        self._thread.start()
        return self

    def read(self, timeout_seconds: float = 1.0) -> FramePacket | None:
        if self._error:
            raise RuntimeError(f"Capture failed for {self.camera_id}") from self._error
        try:
            return self._queue.get(timeout=timeout_seconds)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._stopped.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        capture = self._capture_factory(self.source)
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Could not open source: {self.source}")
            frame_index = 0
            while not self._stopped.is_set():
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if timestamp <= 0:
                    timestamp = time.monotonic()
                packet = FramePacket(self.camera_id, frame_index, timestamp, self._preprocess(frame))
                frame_index += 1
                try:
                    self._queue.put_nowait(packet)
                except queue.Full:
                    _ = self._queue.get_nowait()
                    self._queue.put_nowait(packet)
        except Exception as error:
            self._error = error
        finally:
            capture.release()


def pair_by_timestamp(
    first: Iterable[FramePacket],
    second: Iterable[FramePacket],
    max_delta_seconds: float = 0.15,
) -> Iterator[tuple[FramePacket, FramePacket]]:
    """Yield nearest chronological pairs within the allowed timestamp delta."""
    if max_delta_seconds < 0:
        raise ValueError("max_delta_seconds must be >= 0.")
    first_iterator, second_iterator = iter(first), iter(second)
    try:
        first_packet, second_packet = next(first_iterator), next(second_iterator)
        while True:
            difference = first_packet.timestamp_seconds - second_packet.timestamp_seconds
            if abs(difference) <= max_delta_seconds:
                yield first_packet, second_packet
                first_packet, second_packet = next(first_iterator), next(second_iterator)
            elif difference < 0:
                first_packet = next(first_iterator)
            else:
                second_packet = next(second_iterator)
    except StopIteration:
        return
