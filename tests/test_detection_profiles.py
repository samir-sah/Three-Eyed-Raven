"""CPU-only tests for anonymous detector inference and coordinate merging."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from crowd_tracker.detection_profiles import DetectionProfile, predict_boxes
from crowd_tracker.tiled_detection_cli import tile_windows


def result(rows):
    data = torch.tensor(rows, dtype=torch.float32).reshape(-1, 6)
    return SimpleNamespace(boxes=SimpleNamespace(data=data))


class FakeModel:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def predict(self, frame, **kwargs):
        self.calls.append((frame.shape, kwargs))
        return [self.results[len(self.calls) - 1]]


class DetectionProfileTests(unittest.TestCase):
    def test_profile_validation_and_immutability(self):
        for bad in ("../escape", "with space", "", "x/y", "a" * 97):
            with self.subTest(name=bad), self.assertRaises(ValueError):
                DetectionProfile(bad, "weights.pt", 960)
        for size in (True, 319, 321, 1921, 960.0):
            with self.subTest(imgsz=size), self.assertRaises(ValueError):
                DetectionProfile("safe", "weights.pt", size)
        for tile in (-1, 319, 321, 4097, True):
            with self.subTest(tile=tile), self.assertRaises(ValueError):
                DetectionProfile("safe", "weights.pt", 960, tile_size=tile)
        with self.assertRaises(ValueError):
            DetectionProfile("safe", " ", 960)
        with self.assertRaises(ValueError):
            DetectionProfile("safe", "weights.pt", 960, augment=1)
        profile = DetectionProfile("safe_768", "weights.pt", 1536, tile_size=768)
        with self.assertRaises(AttributeError):
            profile.imgsz = 640

    def test_whole_image_arguments_and_anonymous_rows(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        model = FakeModel([result([[20, 10, 40, 50, .9, 0]])])
        rows = predict_boxes(model, frame, DetectionProfile("whole", "weights.pt", 640))
        self.assertEqual(len(rows[0]), 5)
        self.assertEqual(rows[0][:4], [20, 10, 40, 50])
        self.assertEqual(model.calls[0], (frame.shape, {
            "imgsz": 640, "conf": .001, "iou": .5, "max_det": 300,
            "classes": [0], "device": 0, "verbose": False, "save": False,
            "augment": False, "quantize": 16, "end2end": False,
        }))

    def test_fp32_augmentation_and_end2end_arguments(self):
        model = FakeModel([result([])])
        profile = DetectionProfile("control", "weights.pt", 960, fp16=False, augment=True, end2end=True)
        self.assertEqual(predict_boxes(model, np.zeros((10, 20, 3), np.uint8), profile), [])
        kwargs = model.calls[0][1]
        self.assertEqual(kwargs["quantize"], 32)
        self.assertTrue(kwargs["augment"])
        self.assertTrue(kwargs["end2end"])

    def test_cpu_device_override_with_explicit_fp32_profile(self):
        model = FakeModel([result([[1, 2, 10, 20, .8, 0]])])
        profile = DetectionProfile("cpu", "weights.pt", 640, fp16=False)
        rows = predict_boxes(model, np.zeros((30, 40, 3), np.uint8), profile, device="cpu")
        self.assertEqual(rows[0][:4], [1, 2, 10, 20])
        self.assertEqual(model.calls[0][1]["device"], "cpu")
        self.assertEqual(model.calls[0][1]["quantize"], 32)

    def test_tiles_translate_and_merge_duplicates(self):
        # 640x320 -> a whole frame and tiles at x=0, x=256, x=320.
        windows = tile_windows(640, 320, 320)
        model = FakeModel([
            result([[280, 20, 300, 50, .6, 0]]),
            result([]),
            result([[24, 20, 44, 50, .9, 0]]),
            result([[10, 70, 30, 100, .8, 0]]),
        ])
        rows = predict_boxes(model, np.zeros((320, 640, 3), np.uint8),
                             DetectionProfile("tiled", "weights.pt", 1536, tile_size=320))
        self.assertEqual(windows, [(0, 0, 320, 320), (256, 0, 576, 320), (320, 0, 640, 320)])
        self.assertEqual([row[:4] for row in rows], [[280, 20, 300, 50], [330, 70, 350, 100]])
        self.assertEqual([call[1]["imgsz"] for call in model.calls], [1536, 960, 960, 960])
        self.assertEqual([call[0] for call in model.calls], [(320, 640, 3), *[(320, 320, 3)] * 3])

    def test_internal_borders_rejected_but_image_borders_retained(self):
        model = FakeModel([
            result([]),
            result([[0, 0, 20, 20, .8, 0], [300, 30, 319, 50, .9, 0]]),
            result([[0, 80, 20, 100, .9, 0], [40, 80, 60, 100, .7, 0]]),
            result([[300, 120, 320, 150, .6, 0]]),
        ])
        rows = predict_boxes(model, np.zeros((320, 640, 3), np.uint8),
                             DetectionProfile("borders", "weights.pt", 960, tile_size=320))
        self.assertEqual([row[:4] for row in rows], [[0, 0, 20, 20], [296, 80, 316, 100], [620, 120, 640, 150]])

    def test_internal_horizontal_border_rejection_and_translation(self):
        model = FakeModel([
            result([]),
            result([[10, 300, 40, 319, .9, 0]]),
            result([[10, 0, 40, 20, .8, 0], [10, 20, 40, 40, .7, 0]]),
            result([]),
        ])
        rows = predict_boxes(model, np.zeros((640, 320, 3), np.uint8),
                             DetectionProfile("vertical", "weights.pt", 960, tile_size=320))
        self.assertEqual([row[:4] for row in rows], [[10, 276, 40, 296]])

    def test_clipping_invalid_scores_degenerate_boxes_and_other_class(self):
        model = FakeModel([result([
            [-20, -10, 250, 110, .9, 0],
            [201, 10, 220, 20, .8, 0],
            [10, 10, 10, 20, .7, 0],
            [10, 10, 20, 20, .6, 1],
            [10, 10, 20, 20, float("nan"), 0],
            [10, 10, 20, 20, 1.1, 0],
            [10, 10, 20, 20, -.1, 0],
        ])])
        rows = predict_boxes(model, np.zeros((100, 200, 3), np.uint8),
                             DetectionProfile("clipping", "weights.pt", 640))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][:4], [0, 0, 200, 100])

    def test_tiled_global_max_det_is_300_and_confidence_ordered(self):
        boxes = [[(i % 30) * 5, (i // 30) * 5, (i % 30) * 5 + 2, (i // 30) * 5 + 2,
                  .1 + i / 1000, 0] for i in range(310)]
        model = FakeModel([result(boxes), result([]), result([]), result([])])
        rows = predict_boxes(model, np.zeros((320, 640, 3), np.uint8),
                             DetectionProfile("cap", "weights.pt", 960, tile_size=320))
        self.assertEqual(len(rows), 300)
        self.assertGreater(rows[0][4], rows[-1][4])
        self.assertEqual(rows[0][:4], boxes[-1][:4])

    def test_no_redundant_tile_for_small_image_and_empty_boxes(self):
        model = FakeModel([result([])])
        rows = predict_boxes(model, np.zeros((100, 200, 3), np.uint8),
                             DetectionProfile("small", "weights.pt", 640, tile_size=320))
        self.assertEqual(rows, [])
        self.assertEqual(len(model.calls), 1)
        model = FakeModel([SimpleNamespace(boxes=None)])
        self.assertEqual(predict_boxes(model, np.zeros((100, 200, 3), np.uint8),
                                      DetectionProfile("none", "weights.pt", 640)), [])

    def test_rejects_tracking_output_and_bad_frame(self):
        model = FakeModel([SimpleNamespace(boxes=SimpleNamespace(data=torch.zeros((1, 7))))])
        with self.assertRaises(ValueError):
            predict_boxes(model, np.zeros((100, 200, 3), np.uint8), DetectionProfile("tracked", "weights.pt", 640))
        for frame in (None, np.zeros((10, 20), np.uint8), np.zeros((0, 20, 3), np.uint8)):
            with self.subTest(frame_shape=getattr(frame, "shape", None)), self.assertRaises(ValueError):
                predict_boxes(FakeModel([]), frame, DetectionProfile("invalid", "weights.pt", 640))


if __name__ == "__main__":
    unittest.main()
