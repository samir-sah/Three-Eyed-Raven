import unittest

from crowd_tracker.visdrone_prepare_cli import visdrone_row_to_yolo


class VisDronePreparationTests(unittest.TestCase):
    def test_maps_pedestrian_box_to_person_label(self):
        label = visdrone_row_to_yolo("10,20,40,20,1,1,0,0", 100, 100, {1, 2})
        self.assertEqual(label, "0 0.300000 0.300000 0.400000 0.200000")

    def test_ignores_non_person_and_ignored_annotations(self):
        self.assertIsNone(visdrone_row_to_yolo("10,20,40,20,1,4,0,0", 100, 100, {1, 2}))
        self.assertIsNone(visdrone_row_to_yolo("10,20,40,20,0,1,0,0", 100, 100, {1, 2}))
