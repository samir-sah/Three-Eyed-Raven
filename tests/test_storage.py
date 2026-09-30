import tempfile
import unittest
from pathlib import Path

from crowd_tracker.storage import RunStore


class RunStoreTests(unittest.TestCase):
    def test_upserts_and_lists_completed_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "artifacts" / "demo"
            store = RunStore.for_output_dir(output)
            store.upsert("demo", output, {"run_type": "single_stream", "frames_processed": 10, "processing_fps": 3.5, "alert_count": 2})
            store.upsert("demo", output, {"run_type": "single_stream", "frames_processed": 12, "processing_fps": 4.0, "alert_count": 3})
            runs = store.list_runs()
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["frames_processed"], 12)
            self.assertEqual(runs[0]["alert_count"], 3)
