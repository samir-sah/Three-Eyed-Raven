import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from crowd_tracker.anonymous_cli import count_boxes, validate_config, run, BrowserVideoWriter


class AnonymousTests(unittest.TestCase):
    def test_box_clipping_invalid_values_and_no_identity(self):
        result = count_boxes([[-5, -4, 20, 30, .9], [2, 2, 1, 3, .8], [0, 0, float("nan"), 2, .8], [0, 0, 1, 1, 2]], 10, 15)
        self.assertEqual(result, [[0, 0, 10, 15, .9]])

    def test_config_excludes_identity_and_unsafe_limits(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "video.mp4"
            model = Path(folder) / "best.pt"
            source.touch()
            model.touch()
            config = {"cameras": [{"id": camera, "source": str(source), "model": str(model), "imgsz": 960} for camera in ("aerial", "ground")], "max_frames_per_camera": 3}
            self.assertEqual(validate_config(config), config)
            for key in ("tracker", "reid_checkpoint", "reid_candidate_threshold"):
                with self.assertRaises(ValueError):
                    validate_config({**config, key: "forbidden"})
            for value in (True, 0, -1, 20001):
                with self.assertRaises(ValueError):
                    validate_config({**config, "max_frames_per_camera": value})

    def test_summary_and_partial_failure_preserved_without_identity_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run"
            summary = [{"camera_id": camera, "frames_processed": 3, "peak_count": 0} for camera in ("aerial", "ground")]
            with patch("crowd_tracker.anonymous_cli.validate_config", side_effect=lambda value: value), patch("crowd_tracker.anonymous_cli.analyse_camera", side_effect=summary):
                result = run({"output_dir": str(output), "cameras": [{}, {}]})
            self.assertEqual(result["total_frames_processed"], 6)
            self.assertEqual(result["analysis_type"], "anonymous_detection")
            self.assertFalse((output / "reid_candidates.json").exists())
            with self.assertRaises(FileExistsError):
                with patch("crowd_tracker.anonymous_cli.validate_config", side_effect=lambda value: value):
                    run({"output_dir": str(output)})
            failed = Path(folder) / "failed"
            with patch("crowd_tracker.anonymous_cli.validate_config", side_effect=lambda value: value), patch("crowd_tracker.anonymous_cli.analyse_camera", side_effect=RuntimeError("decode failed")):
                with self.assertRaises(RuntimeError):
                    run({"output_dir": str(failed), "cameras": [{}]})
            self.assertEqual(json.loads((failed / "progress.json").read_text())["status"], "failed")
            self.assertFalse((failed / "summary.json").exists())

    def test_real_browser_encoding_decodes_all_frames(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            writer = BrowserVideoWriter(directory, 65, 49, 10)
            for _ in range(4):
                writer.write(np.zeros((49, 65, 3), dtype=np.uint8))
            writer.close()
            capture = cv2.VideoCapture(str(directory / "browser.mp4"))
            self.assertTrue(capture.isOpened())
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 4)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 64)
            capture.release()


if __name__ == "__main__":
    unittest.main()
