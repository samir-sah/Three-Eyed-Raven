import unittest

from crowd_tracker.ground_detection_cli import match_frame, operating_metrics, parse_annotations, detection_ap


class GroundDetectionTests(unittest.TestCase):
    def test_annotation_scope_discards_identity_and_keeps_occluded_targets(self):
        frames = parse_annotations("1,42,0,0,10,20,1,1,0\n1,99,20,0,10,20,0,7,1")
        self.assertEqual(len(frames[1]["truth"]), 1)
        self.assertEqual(len(frames[1]["ignore"]), 1)
        self.assertNotIn("identity", frames[1])
        self.assertEqual(frames[1]["truth"][0], [-1, -1, 9, 19])

    def test_duplicate_prediction_is_false_positive(self):
        _, hits, neutral, _ = match_frame([[0, 0, 10, 10]], [], [[0, 0, 10, 10, .9], [0, 0, 10, 10, .8]])
        self.assertEqual(hits.tolist(), [True, False])
        self.assertEqual(neutral.tolist(), [False, False])

    def test_valid_person_wins_over_overlapping_ignore_box(self):
        _, hits, neutral, _ = match_frame([[0, 0, 10, 10]], [[0, 0, 20, 20]], [[0, 0, 10, 10, .9], [12, 12, 18, 18, .8]])
        self.assertEqual(hits.tolist(), [True, False])
        self.assertEqual(neutral.tolist(), [False, True])

    def test_threshold_changes_count_and_misses(self):
        row = {"sequence": "sample", "frame": 1, "truth": [[0, 0, 10, 10]], "ignore": [], "predictions": [[0, 0, 10, 10, .2]]}
        high, low = operating_metrics([row], .35), operating_metrics([row], .1)
        self.assertEqual(high["false_negatives"], 1)
        self.assertEqual(high["count_bias"], -1)
        self.assertEqual(low["recall"], 1)
        self.assertEqual(low["count_mae"], 0)

    def test_empty_predictions_and_false_positive_on_empty_frame(self):
        row = {"sequence": "sample", "frame": 1, "truth": [], "ignore": [], "predictions": [[0, 0, 10, 10, .8]]}
        result = operating_metrics([row], .35)
        self.assertEqual(result["false_positives"], 1)
        self.assertEqual(result["count_mae"], 1)
        self.assertEqual(match_frame([], [], [])[1].tolist(), [])

    def test_ap_perfect_and_nonoverlapping_predictions(self):
        row = {"truth": [[0, 0, 10, 10]], "ignore": [], "predictions": [[0, 0, 10, 10, .9]]}
        self.assertGreater(detection_ap([row])["map50"], .99)
        row["predictions"] = [[20, 20, 30, 30, .9]]
        self.assertEqual(detection_ap([row])["map50"], 0)


if __name__ == "__main__":
    unittest.main()
