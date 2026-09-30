import unittest

import numpy as np

from crowd_tracker.acquisition import FramePacket, pair_by_timestamp, preprocess_frame


class AcquisitionTests(unittest.TestCase):
    def test_pairs_nearby_timestamps(self):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        first = [FramePacket("a", 0, 0.0, frame), FramePacket("a", 1, 1.0, frame)]
        second = [FramePacket("b", 0, 0.05, frame), FramePacket("b", 1, 1.05, frame)]
        self.assertEqual(len(list(pair_by_timestamp(first, second, 0.1))), 2)

    def test_skips_unmatched_packet(self):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        first = [FramePacket("a", 0, 0.0, frame), FramePacket("a", 1, 1.0, frame)]
        second = [FramePacket("b", 0, 0.6, frame), FramePacket("b", 1, 1.02, frame)]
        pairs = list(pair_by_timestamp(first, second, 0.1))
        self.assertEqual([(a.frame_index, b.frame_index) for a, b in pairs], [(1, 1)])

    def test_preprocess_resizes_frame(self):
        frame = np.zeros((10, 20, 3), dtype=np.uint8)
        self.assertEqual(preprocess_frame(frame, size=(8, 6)).shape, (6, 8, 3))
