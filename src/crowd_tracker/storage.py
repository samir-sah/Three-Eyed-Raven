"""Small local SQLite index for completed tracking runs."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class RunStore:
    """Persists compact run metadata while artifacts remain on disk."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    output_dir TEXT NOT NULL,
                    run_type TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    frames_processed INTEGER NOT NULL,
                    processing_fps REAL NOT NULL,
                    alert_count INTEGER NOT NULL,
                    summary_json TEXT NOT NULL
                )
            """)
            connection.commit()
        finally:
            connection.close()

    @classmethod
    def for_output_dir(cls, output_dir: Path) -> "RunStore":
        configured = os.environ.get("CROWD_TRACKER_DB")
        return cls(Path(configured) if configured else output_dir.parent / "runs.sqlite3")

    def upsert(self, run_id: str, output_dir: Path, summary: dict[str, Any]) -> None:
        frames = int(summary.get("total_frames_processed", summary.get("frames_processed", 0)))
        fps = float(summary.get("aggregate_processing_fps", summary.get("processing_fps", 0)))
        connection = self._connect()
        try:
            connection.execute("""
                INSERT INTO runs (run_id, output_dir, run_type, completed_at, frames_processed, processing_fps, alert_count, summary_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    output_dir = excluded.output_dir,
                    run_type = excluded.run_type,
                    completed_at = excluded.completed_at,
                    frames_processed = excluded.frames_processed,
                    processing_fps = excluded.processing_fps,
                    alert_count = excluded.alert_count,
                    summary_json = excluded.summary_json
            """, (
                run_id,
                str(output_dir),
                str(summary.get("run_type", "single_stream")),
                datetime.now(UTC).isoformat(),
                frames,
                fps,
                int(summary.get("alert_count", 0)),
                json.dumps(summary, separators=(",", ":")),
            ))
            connection.commit()
        finally:
            connection.close()

    def list_runs(self) -> list[dict[str, Any]]:
        connection = self._connect()
        try:
            rows = connection.execute("""
                SELECT run_id, output_dir, run_type, completed_at, frames_processed, processing_fps, alert_count
                FROM runs ORDER BY completed_at DESC
            """).fetchall()
        finally:
            connection.close()
        return [dict(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection
