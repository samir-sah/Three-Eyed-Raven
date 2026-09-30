import unittest

from crowd_tracker.config import AppConfig, ZoneConfig


class ConfigTests(unittest.TestCase):
    def test_rejects_duplicate_zone_ids(self):
        config = AppConfig(
            source="video.mp4",
            output_dir="artifacts",
            zones=[
                ZoneConfig("gate", [[0, 0], [1, 0], [0, 1]], 1),
                ZoneConfig("gate", [[2, 2], [3, 2], [2, 3]], 1),
            ],
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            config.validate()

    def test_rejects_invalid_confidence(self):
        config = AppConfig(source="video.mp4", output_dir="artifacts", confidence_threshold=1.2)
        with self.assertRaisesRegex(ValueError, "confidence"):
            config.validate()

    def test_rejects_negative_reconnect_settings(self):
        config = AppConfig(source="rtsp://camera", output_dir="artifacts", reconnect_attempts=-1)
        with self.assertRaisesRegex(ValueError, "reconnect_attempts"):
            config.validate()
