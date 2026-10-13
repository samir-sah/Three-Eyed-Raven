import unittest
from crowd_tracker.score_target_cli import choose, raw_counts, profiles
from pathlib import Path


class ScoreTargetTests(unittest.TestCase):
    def test_selection_uses_calibration_ap_not_precision_or_count_only(self):
        first = {"name": "higher_ap", "detection": {"map50": .72}, "fixed": {"count_mae": 5}, "speed": {"images_per_second": 10}}
        second = {"name": "lower_ap", "detection": {"map50": .69}, "fixed": {"count_mae": 2}, "speed": {"images_per_second": 20}}
        self.assertEqual(choose([first, second])["name"], "higher_ap")
        second["detection"]["map50"] = .72
        self.assertEqual(choose([first, second])["name"], "lower_ap")

    def test_raw_count_keeps_predictions_on_ignored_regions(self):
        rows = [{"truth": [[0, 0, 10, 10]], "ignore": [[20, 20, 30, 30]], "predictions": [[0, 0, 10, 10, .9], [20, 20, 30, 30, .8]]}]
        self.assertEqual(raw_counts(rows, .35)["mae"], 1)

    def test_candidate_names_unique_and_all_anonymous_non_e2e(self):
        candidates = profiles(Path("root"))
        self.assertEqual(len({item.name for item in candidates}), len(candidates))
        self.assertFalse(any(item.end2end for item in candidates))
        self.assertFalse(candidates[0].fp16)

    def test_advanced_suite_is_fixed_and_contains_only_declared_profiles(self):
        candidates = profiles(Path("root"), "advanced")
        self.assertEqual(len(candidates), 18)
        self.assertEqual(len({item.name for item in candidates}), 18)
        self.assertTrue(all(item.augment for item in candidates[1:]))
        self.assertFalse(any(item.end2end for item in candidates))
        self.assertTrue(all(item.imgsz <= 1920 for item in candidates))
        with self.assertRaises(ValueError):
            profiles(Path("root"), "unknown")


if __name__ == "__main__":
    unittest.main()
