import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class MotImportTests(unittest.TestCase):
    def test_imports_person_rows_and_converts_box_coordinates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "gt.txt"
            destination = root / "ground_truth.jsonl"
            source.write_text("1,7,10,20,30,40,1,1,0.8\n2,8,0,0,5,5,1,2,1\n", encoding="utf-8")
            result = subprocess.run([
                sys.executable, "-m", "crowd_tracker.mot_import_cli", "--input", str(source), "--output", str(destination),
            ], check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(result.stdout)["written"], 1)
            record = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(record["frame_index"], 0)
            self.assertEqual(record["bbox_xyxy"], [10.0, 20.0, 40.0, 60.0])
