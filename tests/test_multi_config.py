import unittest

from crowd_tracker.config import CameraStreamConfig, MultiStreamConfig


class MultiStreamConfigTests(unittest.TestCase):
    def test_rejects_single_camera_configuration(self):
        config = MultiStreamConfig(
            output_dir="artifacts/two_stream_demo",
            cameras=[CameraStreamConfig(id="aerial_1", source="aerial.mp4")],
        )
        with self.assertRaisesRegex(ValueError, "at least two"):
            config.validate()

    def test_rejects_duplicate_camera_ids(self):
        config = MultiStreamConfig(
            output_dir="artifacts/two_stream_demo",
            cameras=[
                CameraStreamConfig(id="camera", source="a.mp4"),
                CameraStreamConfig(id="camera", source="b.mp4"),
            ],
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            config.validate()
