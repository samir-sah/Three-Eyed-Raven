import unittest
from crowd_tracker.tiled_detection_cli import axis_starts, tile_windows, label_boxes


class TileGeometryTests(unittest.TestCase):
    def test_small_image_is_not_padded_with_extra_tiles(self):
        self.assertEqual(tile_windows(100, 80, 640), [(0, 0, 100, 80)])

    def test_edges_covered_without_duplicate_tiles(self):
        windows = tile_windows(1920, 1080, 640)
        self.assertEqual(len(windows), len(set(windows)))
        for x in range(1920):
            self.assertTrue(any(a <= x < c for a, b, c, d in windows))
        for y in range(1080):
            self.assertTrue(any(b <= y < d for a, b, c, d in windows))
        self.assertIn((1280, 440, 1920, 1080), windows)

    def test_normalized_label_conversion_and_empty_labels(self):
        self.assertEqual(label_boxes("0 .5 .5 .2 .4", 100, 200), [[40, 60, 60, 140]])
        self.assertEqual(label_boxes("", 100, 100), [])

    def test_invalid_geometry_fails(self):
        with self.assertRaises(ValueError):
            axis_starts(100, 0)

    def test_official_matcher_does_not_count_duplicate_predictions_twice(self):
        import torch
        from ultralytics.engine.validator import BaseValidator

        # Test the exact official matcher used by the evaluator, with no GPU.
        class Matcher:
            iouv = torch.linspace(.5, .95, 10)
        matched = BaseValidator.match_predictions(Matcher(), torch.zeros(2), torch.zeros(1), torch.ones((1, 2)))
        self.assertEqual(tuple(matched.shape), (2, 10))
        self.assertEqual(matched[:, 0].sum().item(), 1)

    def test_official_matcher_rejects_nonoverlap(self):
        import torch
        from ultralytics.engine.validator import BaseValidator

        class Matcher:
            iouv = torch.linspace(.5, .95, 10)
        matched = BaseValidator.match_predictions(Matcher(), torch.zeros(1), torch.zeros(1), torch.zeros((1, 1)))
        self.assertFalse(matched.any().item())
