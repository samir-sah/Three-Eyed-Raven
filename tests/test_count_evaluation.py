import unittest

from crowd_tracker.count_evaluation import count_metrics, evaluate_counts, partition_for_image


class CountEvaluationTests(unittest.TestCase):
    def test_count_errors_include_false_positives_in_empty_images(self):
        result = count_metrics([
            {"actual": 0, "scores": [0.9, 0.8]},
            {"actual": 4, "scores": [0.9, 0.2]},
        ], 0.5)
        self.assertEqual(result["mae"], 2.5)
        self.assertAlmostEqual(result["rmse"], (6.5) ** 0.5)
        self.assertEqual(result["bias"], -0.5)

    def test_same_sequence_never_crosses_calibration_report_boundary(self):
        self.assertEqual(partition_for_image("0000071_00012.jpg"), partition_for_image("0000071_98765.jpg"))

    def test_report_labels_do_not_influence_threshold_selection(self):
        names = [f"{index:07}_00001.jpg" for index in range(40)]
        rows = [{"image": name, "actual": 1, "scores": [0.9, 0.15]} for name in names]
        first = evaluate_counts(rows)
        changed = [{**row, "actual": 100 if partition_for_image(row["image"]) == "report" else row["actual"]} for row in rows]
        second = evaluate_counts(changed)
        self.assertEqual(first["selected_threshold"], second["selected_threshold"])
        self.assertNotEqual(first["report"]["mae"], second["report"]["mae"])

    def test_empty_partition_fails_instead_of_reporting_zero_error(self):
        with self.assertRaises(ValueError):
            evaluate_counts([{"image": "0000001_00001.jpg", "actual": 0, "scores": []}])


if __name__ == "__main__":
    unittest.main()
