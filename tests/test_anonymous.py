import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from crowd_tracker.anonymous_cli import analyse_camera, count_boxes, validate_config, run, BrowserVideoWriter


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

    def test_high_resolution_requires_anonymous_profile_matching_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "fixture"
            file.touch()
            profile = {"name": "validated_tta", "checkpoint": str(file), "imgsz": 1536, "augment": True}
            cameras = [{"id": camera, "source": str(file), "model": str(file), "imgsz": 1536, "detection_profile": profile} for camera in ("aerial", "ground")]
            self.assertEqual(validate_config({"cameras": cameras})["cameras"], cameras)
            cameras[0] = {**cameras[0], "imgsz": 960}
            with self.assertRaises(ValueError):
                validate_config({"cameras": cameras})
            cameras[0] = {**cameras[0], "imgsz": 1536, "detection_profile": None}
            with self.assertRaises(ValueError):
                validate_config({"cameras": cameras})

    def test_profile_checkpoint_must_match_camera_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            source, model, other = (Path(folder) / name for name in ("video.mp4", "measured.pt", "different.pt"))
            for file in (source, model, other):
                file.touch()
            profile = {"name": "measured_960", "checkpoint": str(other), "imgsz": 960}
            cameras = [{"id": camera, "source": str(source), "model": str(model), "imgsz": 960,
                        "detection_profile": profile} for camera in ("aerial", "ground")]
            with self.assertRaisesRegex(ValueError, "match the camera checkpoint"):
                validate_config({"cameras": cameras})

    def _mock_profile_camera(self, predictions, *, device="auto", cuda_available=False,
                             marker_demo=None, marker_session=None, encoder_error=None):
        """Exercise real annotation/count code without a model, codec or GPU call."""
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "output"
            output.mkdir()
            source, checkpoint = root / "video.mp4", root / "measured.pt"
            source.touch()
            checkpoint.touch()
            profile = {"name": "measured_tta", "checkpoint": str(checkpoint), "imgsz": 1536,
                       "tile_size": 0, "augment": True, "fp16": True, "end2end": False}
            camera = {"id": "ground", "source": str(source), "model": str(checkpoint), "imgsz": 1536,
                      "detection_profile": profile}
            config = {"device": device, "confidence_threshold": .35, "max_frames_per_camera": len(predictions)}
            if marker_demo is not None:
                config["marker_demo"] = marker_demo
            read_index = 0

            def read_frame():
                nonlocal read_index
                read_index += 1
                return True, np.zeros((80, 100, 3), dtype=np.uint8)

            def capture_property(key):
                return {cv2.CAP_PROP_FPS: 10, cv2.CAP_PROP_FRAME_WIDTH: 100,
                        cv2.CAP_PROP_FRAME_HEIGHT: 80, cv2.CAP_PROP_POS_MSEC: read_index * 100}[key]

            capture = Mock()
            capture.isOpened.return_value = True
            capture.read.side_effect = read_frame
            capture.get.side_effect = capture_property
            frames = []
            writer = Mock(size=(100, 80))
            writer.write.side_effect = lambda frame: frames.append(frame.copy())
            if encoder_error is not None:
                writer.close.side_effect = encoder_error
            model = SimpleNamespace(names={0: "person"}, predict=Mock())
            constructor = Mock(return_value=model)
            fake_ultralytics = SimpleNamespace(YOLO=constructor)
            fake_torch = SimpleNamespace(cuda=SimpleNamespace(
                is_available=Mock(return_value=cuda_available), empty_cache=Mock()))
            with patch.dict(sys.modules, {"ultralytics": fake_ultralytics, "torch": fake_torch}), \
                 patch("cv2.VideoCapture", return_value=capture), \
                 patch("cv2.rectangle", wraps=cv2.rectangle) as rectangles, \
                 patch("crowd_tracker.marker_session.MarkerSession", return_value=marker_session), \
                 patch("crowd_tracker.anonymous_cli.BrowserVideoWriter", return_value=writer), \
                 patch("crowd_tracker.anonymous_cli.predict_boxes", side_effect=predictions) as predict:
                summary = analyse_camera(camera, config, output, output / "progress.json")
            rows = [json.loads(line) for line in (output / "ground" / "counts.jsonl").read_text().splitlines()]
            files = [file.name for file in (output / "ground").iterdir()]
            constructor.assert_called_once_with(str(checkpoint))
            model.predict.assert_not_called()
            capture.release.assert_called_once()
            writer.close.assert_called_once_with(False)
            self.assertEqual(capture.read.call_count, len(predictions))
            self.assertEqual(camera["detection_profile"], profile)
            return summary, rows, frames, predict.call_args_list, rectangles.call_args_list, files

    def test_profile_counts_and_exported_boxes_share_the_point35_filter(self):
        predictions = [
            [[5, 50, 25, 70, .9], [30, 50, 40, 60, .35], [50, 50, 60, 60, .34999],
             [-1, 50, 10, 90, .8], [4, 4, 3, 5, .99], [0, 0, 1, 1, 1.5]],
            [],
            [[60, 50, 70, 60, .35]],
        ]
        summary, rows, frames, calls, rectangles, files = self._mock_profile_camera(predictions)
        self.assertEqual([row["count"] for row in rows], [3, 0, 1])
        self.assertEqual(rows[0]["boxes_scores"], [[5, 50, 25, 70, .9], [30, 50, 40, 60, .35], [0, 50, 10, 80, .8]])
        self.assertEqual(rows[1]["boxes_scores"], [])
        self.assertEqual(rows[2]["boxes_scores"], [[60, 50, 70, 60, .35]])
        self.assertTrue(all(row["count"] == len(row["boxes_scores"]) for row in rows))
        self.assertEqual([row["frame_index"] for row in rows], [0, 1, 2])
        self.assertEqual([row["timestamp_seconds"] for row in rows], [0, .1, .2])
        self.assertEqual([row["source_timestamp_seconds"] for row in rows], [.1, .2, .3])
        self.assertEqual(summary["frames_processed"], len(frames))
        self.assertEqual(summary["peak_count"], 3)
        self.assertEqual(summary["mean_count"], 4 / 3)
        self.assertEqual(summary["duration_seconds"], .3)
        exported = [(call.args[1], call.args[2]) for call in rectangles
                    if (call.args[1], call.args[2]) != ((8, 8), (220, 43))]
        self.assertEqual(exported, [((5, 50), (25, 70)), ((30, 50), (40, 60)),
                                    ((0, 50), (10, 80)), ((60, 50), (70, 60))])
        self.assertEqual(files, ["counts.jsonl"])
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(len(box) == 5 for row in rows for box in row["boxes_scores"]))

    def test_auto_cpu_profile_uses_fp32_and_preserves_approved_input(self):
        summary, _, _, calls, _, _ = self._mock_profile_camera([[]], cuda_available=False)
        runtime = calls[0].args[2]
        self.assertFalse(runtime.fp16)
        self.assertTrue(runtime.augment)
        self.assertEqual(runtime.imgsz, 1536)
        self.assertEqual(calls[0].kwargs, {"device": "cpu"})
        self.assertEqual(summary["runtime_device"], "cpu")
        self.assertFalse(summary["runtime_detection_profile"]["fp16"])

    def test_explicit_cpu_keeps_fp32_even_when_cuda_is_available(self):
        summary, _, _, calls, _, _ = self._mock_profile_camera([[]], device="cpu", cuda_available=True)
        self.assertFalse(calls[0].args[2].fp16)
        self.assertEqual(calls[0].kwargs, {"device": "cpu"})
        self.assertEqual(summary["runtime_device"], "cpu")

    def test_cuda_profile_keeps_measured_fp16_without_executing_gpu(self):
        summary, _, _, calls, _, _ = self._mock_profile_camera([[]], cuda_available=True)
        self.assertTrue(calls[0].args[2].fp16)
        self.assertEqual(calls[0].kwargs, {"device": 0})
        self.assertTrue(summary["runtime_detection_profile"]["fp16"])
        self.assertEqual(summary["runtime_device"], "0")

    def test_opted_in_marker_scan_is_separate_from_detection_counts(self):
        session = Mock()
        session.read.return_value = [{"code": "V07", "corners": [[10, 50], [20, 50], [20, 60], [10, 60]]}]
        session.summary.return_value = {"frames_processed": 1, "frames_scanned": 1,
                                        "read_failures": 0, "observations": {}, "duplicate_codes": []}
        config = {"enabled": True, "confirmed_volunteer_demo": True,
                  "allowed_codes": ["V07"], "min_reads": 3}
        summary, rows, _, _, _, _ = self._mock_profile_camera([[]], marker_demo=config, marker_session=session)
        self.assertEqual(rows[0]["count"], 0)
        self.assertEqual(rows[0]["boxes_scores"], [])
        self.assertEqual(summary["markers"], session.summary.return_value)
        session.read.assert_called_once()
        self.assertEqual(session.read.call_args.args[1:], (0, 0))
        session.summary.assert_called_once_with(1)
        session.close.assert_called_once()

    def test_marker_log_is_closed_even_if_encoder_finalization_fails(self):
        session = Mock()
        session.read.return_value = []
        config = {"enabled": True, "confirmed_volunteer_demo": True, "allowed_codes": ["V07"]}
        with self.assertRaisesRegex(RuntimeError, "encode failed"):
            self._mock_profile_camera([[]], marker_demo=config, marker_session=session,
                                      encoder_error=RuntimeError("encode failed"))
        session.close.assert_called_once()

    def test_disabled_marker_mode_never_reads_tags(self):
        session = Mock()
        summary, _, _, _, _, files = self._mock_profile_camera([[]], marker_demo={"enabled": False}, marker_session=session)
        session.read.assert_not_called()
        session.close.assert_not_called()
        self.assertNotIn("markers", summary)
        self.assertEqual(files, ["counts.jsonl"])

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
