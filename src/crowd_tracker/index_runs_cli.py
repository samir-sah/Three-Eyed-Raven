"""Index existing artifact summaries into the local SQLite run store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .storage import RunStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Index completed artifact runs in SQLite.")
    parser.add_argument("--artifacts", default="artifacts", help="Artifact root to scan.")
    args = parser.parse_args()
    root = Path(args.artifacts)
    store = RunStore.for_output_dir(root / "placeholder")
    indexed = 0
    for directory in root.iterdir() if root.exists() else []:
        if not directory.is_dir():
            continue
        try:
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(summary, dict):
            summary["alert_count"] = _alert_count(directory, summary)
            store.upsert(directory.name, directory, summary)
            indexed += 1
    print(json.dumps({"indexed": indexed, "database": str(store.path)}, indent=2))


def _alert_count(directory: Path, summary: dict) -> int:
    if summary.get("run_type") == "multi_stream":
        return sum(_jsonl_count(directory / str(camera.get("camera_id")) / "alerts.jsonl") for camera in summary.get("cameras", []))
    return _jsonl_count(directory / "alerts.jsonl")


def _jsonl_count(path: Path) -> int:
    try:
        return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())
    except OSError:
        return 0


if __name__ == "__main__":
    main()
