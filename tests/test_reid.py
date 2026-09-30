import unittest

import numpy as np

from crowd_tracker.reid import CrossCameraMatcher, TrackDescriptor


class ReIdTests(unittest.TestCase):
    def test_ranks_only_cross_camera_descriptors(self):
        first = TrackDescriptor("aerial", 1, 0.0, 1.0, 2, np.array([1.0, 0.0], dtype=np.float32))
        second = TrackDescriptor("cctv", 4, 0.0, 1.0, 2, np.array([0.99, 0.01], dtype=np.float32))
        third = TrackDescriptor("cctv", 5, 0.0, 1.0, 2, np.array([0.0, 1.0], dtype=np.float32))
        results = CrossCameraMatcher(candidate_threshold=0.9, top_k=1).rank([first, second, third])
        aerial_result = next(item for item in results if item["probe_camera_id"] == "aerial")
        self.assertEqual(aerial_result["gallery_track_id"], 4)
        self.assertEqual(aerial_result["status"], "review")

    def test_never_compares_a_track_to_same_camera(self):
        first = TrackDescriptor("aerial", 1, 0.0, 1.0, 2, np.array([1.0, 0.0], dtype=np.float32))
        second = TrackDescriptor("aerial", 2, 0.0, 1.0, 2, np.array([1.0, 0.0], dtype=np.float32))
        self.assertEqual(CrossCameraMatcher().rank([first, second]), [])
