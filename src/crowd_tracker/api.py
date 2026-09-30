"""Local FastAPI service exposing completed tracking artifacts.

This is deliberately read-only: camera processing writes artifacts, while the API
provides a stable boundary for dashboards and future authenticated operators.
"""

from __future__ import annotations

import json
import os
import re
import secrets
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader

from .storage import RunStore


_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class ArtifactRepository:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def list_runs(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        runs = []
        for directory in self.root.iterdir():
            if not directory.is_dir() or not _SAFE_RUN_ID.fullmatch(directory.name):
                continue
            summary = self._read_json(directory / "summary.json")
            if not summary:
                continue
            runs.append({
                "id": directory.name,
                "run_type": summary.get("run_type", "single_stream"),
                "updated_at": (directory / "summary.json").stat().st_mtime,
                "frames_processed": summary.get("total_frames_processed", summary.get("frames_processed", 0)),
                "processing_fps": summary.get("aggregate_processing_fps", summary.get("processing_fps", 0)),
                "camera_count": len(summary.get("cameras", [])) or 1,
            })
        return sorted(runs, key=lambda run: run["updated_at"], reverse=True)

    def read_run(self, run_id: str) -> dict[str, Any] | None:
        if not _SAFE_RUN_ID.fullmatch(run_id):
            return None
        directory = self.root / run_id
        summary = self._read_json(directory / "summary.json")
        if not summary:
            return None
        if summary.get("run_type") == "multi_stream":
            cameras = []
            total_alerts = 0
            for camera in summary.get("cameras", []):
                camera_id = camera.get("camera_id")
                if not isinstance(camera_id, str) or not _SAFE_RUN_ID.fullmatch(camera_id):
                    continue
                alerts = self._read_json_lines(directory / camera_id / "alerts.jsonl")
                total_alerts += len(alerts)
                cameras.append({**camera, "alert_count": len(alerts), "alerts": alerts[-10:]})
            return {
                "id": run_id,
                "summary": summary,
                "cameras": cameras,
                "alert_count": total_alerts,
                "reid": self._read_json(directory / "reid_candidates.json") or {"summary": {}, "candidates": []},
            }
        alerts = self._read_json_lines(directory / "alerts.jsonl")
        return {"id": run_id, "summary": summary, "alert_count": len(alerts), "alerts": alerts[-10:]}

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _read_json_lines(path: Path) -> list[dict[str, Any]]:
        try:
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, json.JSONDecodeError):
            return []


def create_app(artifacts_root: Path | None = None, api_key: str | None = None) -> FastAPI:
    root = artifacts_root or Path(os.environ.get("CROWD_TRACKER_ARTIFACTS", "artifacts"))
    repository = ArtifactRepository(root)
    configured_key = api_key if api_key is not None else os.environ.get("CROWD_TRACKER_API_KEY")
    key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
    app = FastAPI(title="Three Eyed Raven Run Service", version="0.1.0")

    def require_api_key(value: str | None = Security(key_header)) -> None:
        if configured_key and not (value and secrets.compare_digest(value, configured_key)):
            raise HTTPException(status_code=401, detail="Valid X-API-Key required")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "artifacts_root": str(repository.root), "run_count": len(repository.list_runs())}

    @app.get("/runs", dependencies=[Depends(require_api_key)])
    def runs() -> dict[str, list[dict[str, Any]]]:
        return {"runs": repository.list_runs()}

    @app.get("/runs/{run_id}", dependencies=[Depends(require_api_key)])
    def run_detail(run_id: str) -> dict[str, Any]:
        run = repository.read_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return run

    @app.get("/history", dependencies=[Depends(require_api_key)])
    def history() -> dict[str, list[dict[str, Any]]]:
        return {"runs": RunStore.for_output_dir(root / "placeholder").list_runs()}

    return app


app = create_app()
