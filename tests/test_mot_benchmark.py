import unittest

from crowd_tracker.mot_benchmark_cli import aggregate_metrics


class MotBenchmarkTests(unittest.TestCase):
    def test_aggregate_metrics_combines_counts_and_weights_hota(self):
        summary = aggregate_metrics([
            {"ground_truth_boxes": 10, "predicted_boxes": 10, "matches": 8, "false_positives": 2, "false_negatives": 2, "id_switches": 1, "id_true_positives": 7, "id_false_positives": 3, "id_false_negatives": 3, "motp": 0.8, "hota": 0.6},
            {"ground_truth_boxes": 20, "predicted_boxes": 20, "matches": 18, "false_positives": 2, "false_negatives": 2, "id_switches": 1, "id_true_positives": 17, "id_false_positives": 3, "id_false_negatives": 3, "motp": 0.9, "hota": 0.9},
        ])
        self.assertEqual(summary["matches"], 26)
        self.assertEqual(summary["mota"], 0.6667)
        self.assertEqual(summary["idf1"], 0.8)
        self.assertEqual(summary["hota"], 0.8)
        self.assertEqual(summary["motp"], 0.8692)
