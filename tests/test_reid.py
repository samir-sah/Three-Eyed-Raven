import unittest

import numpy as np

from crowd_tracker.reid import CrossCameraMatcher, TrackDescriptor
from crowd_tracker.reid_evaluation import LabeledEmbedding, evaluate_retrieval


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

    def test_retrieval_metrics_report_perfect_cross_camera_ranking(self):
        queries = [
            LabeledEmbedding("a1", "1", "aerial", np.array([1.0, 0.0], dtype=np.float32)),
            LabeledEmbedding("a2", "2", "aerial", np.array([0.0, 1.0], dtype=np.float32)),
        ]
        gallery = [
            LabeledEmbedding("g1", "1", "ground", np.array([1.0, 0.0], dtype=np.float32)),
            LabeledEmbedding("g2", "2", "ground", np.array([0.0, 1.0], dtype=np.float32)),
        ]
        result = evaluate_retrieval(queries, gallery)
        self.assertEqual(result["rank_1"], 1.0)
        self.assertEqual(result["mAP"], 1.0)
