import argparse
import unittest

from crowd_tracker.detection_sweep_cli import valid_sizes


class ResolutionSweepTests(unittest.TestCase):
    def test_valid_resolutions_preserve_order(self):
        self.assertEqual(valid_sizes("960,1280,1536,1920,2240"), [960, 1280, 1536, 1920, 2240])

    def test_reject_duplicate_off_stride_and_unbounded_sizes(self):
        for value in ("960,960", "961", "288", "2592"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                valid_sizes(value)
