"""Low-overhead per-stage latency collection for pipeline run summaries."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager


class StageProfiler:
    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def measure(self, stage: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self._samples[stage].append((time.perf_counter() - started) * 1000)

    def summary(self) -> list[dict]:
        result = []
        for stage, samples in sorted(self._samples.items()):
            ordered = sorted(samples)
            index = min(len(ordered) - 1, int(len(ordered) * 0.95))
            result.append({
                "stage": stage,
                "calls": len(samples),
                "mean_ms": round(sum(samples) / len(samples), 3),
                "p95_ms": round(ordered[index], 3),
                "total_ms": round(sum(samples), 3),
            })
        return result
