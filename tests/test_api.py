import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from crowd_tracker.api import create_app


class ApiTests(unittest.TestCase):
    def test_lists_and_reads_multi_camera_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "demo_run"
            (run / "camera_1").mkdir(parents=True)
            (run / "summary.json").write_text(json.dumps({
                "run_type": "multi_stream",
                "total_frames_processed": 12,
                "aggregate_processing_fps": 5.5,
                "cameras": [{"camera_id": "camera_1", "frames_processed": 12}],
            }), encoding="utf-8")
            (run / "camera_1" / "alerts.jsonl").write_text(json.dumps({"zone_id": "gate", "count": 3}) + "\n", encoding="utf-8")
            client = TestClient(create_app(root))

            self.assertEqual(client.get("/health").json()["run_count"], 1)
            listing = client.get("/runs").json()["runs"]
            self.assertEqual(listing[0]["id"], "demo_run")
            detail = client.get("/runs/demo_run").json()
            self.assertEqual(detail["alert_count"], 1)
            self.assertEqual(detail["cameras"][0]["alerts"][0]["zone_id"], "gate")

    def test_unknown_or_unsafe_run_returns_not_found(self):
        with tempfile.TemporaryDirectory() as temporary:
            client = TestClient(create_app(Path(temporary)))
            self.assertEqual(client.get("/runs/missing").status_code, 404)
            self.assertEqual(client.get("/runs/..%2Fsecret").status_code, 404)
