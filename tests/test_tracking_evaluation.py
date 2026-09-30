import unittest

from crowd_tracker.tracking_evaluation import LabeledBox, evaluate_tracking


class TrackingEvaluationTests(unittest.TestCase):
    def test_perfect_tracking_scores_are_one(self):
        ground_truth = [
            LabeledBox(0, "a", (0, 0, 10, 10)),
            LabeledBox(1, "a", (1, 0, 11, 10)),
        ]
        predictions = [
            LabeledBox(0, "local_1", (0, 0, 10, 10)),
            LabeledBox(1, "local_1", (1, 0, 11, 10)),
        ]
        metrics = evaluate_tracking(ground_truth, predictions)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["mota"], 1.0)
        self.assertEqual(metrics["id_switches"], 0)
        self.assertEqual(metrics["idf1"], 1.0)
        self.assertEqual(metrics["hota"], 1.0)

    def test_identity_switch_and_false_positive_are_counted(self):
        ground_truth = [
            LabeledBox(0, "a", (0, 0, 10, 10)),
            LabeledBox(1, "a", (1, 0, 11, 10)),
        ]
        predictions = [
            LabeledBox(0, "local_1", (0, 0, 10, 10)),
            LabeledBox(1, "local_2", (1, 0, 11, 10)),
            LabeledBox(1, "extra", (30, 0, 40, 10)),
        ]
        metrics = evaluate_tracking(ground_truth, predictions)
        self.assertEqual(metrics["id_switches"], 1)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["mota"], 0.0)
        self.assertEqual(metrics["idf1"], 0.4)

    def test_global_identity_assignment_penalises_fragmented_tracks(self):
        ground_truth = [
            LabeledBox(0, "a", (0, 0, 10, 10)),
            LabeledBox(1, "a", (0, 0, 10, 10)),
            LabeledBox(0, "b", (20, 0, 30, 10)),
            LabeledBox(1, "b", (20, 0, 30, 10)),
        ]
        predictions = [
            LabeledBox(0, "local_1", (0, 0, 10, 10)),
            LabeledBox(1, "local_2", (0, 0, 10, 10)),
            LabeledBox(0, "local_3", (20, 0, 30, 10)),
            LabeledBox(1, "local_3", (20, 0, 30, 10)),
        ]
        metrics = evaluate_tracking(ground_truth, predictions)
        self.assertEqual(metrics["id_true_positives"], 3)
        self.assertEqual(metrics["idf1"], 0.75)
        self.assertAlmostEqual(metrics["hota"], 0.866, places=3)
