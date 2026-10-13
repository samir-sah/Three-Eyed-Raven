import unittest
from crowd_tracker.ground_training_cli import yolo_labels, TRAIN, VALIDATION
from crowd_tracker.ground_detection_cli import REPORT


class GroundTrainingTests(unittest.TestCase):
    def test_scene_partitions_do_not_overlap(self):
        self.assertFalse(TRAIN & VALIDATION or TRAIN & REPORT or VALIDATION & REPORT)

    def test_labels_are_clipped_normalized_and_anonymous(self):
        text = yolo_labels([[-10, -10, 100, 200], [20, 20, 10, 10]], 100, 200)
        self.assertEqual(text, "0 0.50000000 0.50000000 1.00000000 1.00000000\n")
        self.assertEqual(yolo_labels([], 100, 200), "")


if __name__ == "__main__":
    unittest.main()
