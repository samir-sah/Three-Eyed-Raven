import unittest

from crowd_tracker.preflight_cli import redact_source


class PreflightTests(unittest.TestCase):
    def test_redacts_stream_credentials(self):
        source = "rtsp://operator:private-password@camera.example:554/live"
        self.assertEqual(redact_source(source), "rtsp://***@camera.example:554/live")

    def test_keeps_local_video_path_unchanged(self):
        self.assertEqual(redact_source("data/demo.mp4"), "data/demo.mp4")
