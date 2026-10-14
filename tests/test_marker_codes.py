import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from crowd_tracker.marker_codes import (
    MarkerReader, generate_qr_image, normalize_codes, summarize_marker_cameras, validate_marker_config,
)


def corners(x=10, y=10, size=30):
    return [[x, y], [x + size, y], [x + size, y + size], [x, y + size]]


def summary(camera, observations, duplicate_codes=None):
    return {"camera_id": camera, "markers": {
        "frames_processed": 100, "frames_scanned": 100, "read_failures": 0,
        "observations": {code: {"first_frame": 0, "last_frame": 90, "first_seconds": 0,
                                "last_seconds": 9, "read_frames": reads, "evidence_file": f"{code}.jpg"}
                         for code, reads in observations.items()},
        "duplicate_codes": duplicate_codes or [],
    }}


class MarkerCodeTests(unittest.TestCase):
    def test_codes_are_strict_distinct_and_bounded(self):
        self.assertEqual(normalize_codes(["V99", "V01", "V01"]), ["V01", "V99"])
        for bad in ["V00", "V100", "V1", "v01", " V01", "V01\n", "TER:V01", "https://example.com", "name", 1]:
            with self.subTest(code=bad), self.assertRaises(ValueError):
                normalize_codes([bad])
        with self.assertRaises(ValueError):
            normalize_codes("V01")

    def test_config_is_off_by_default_and_requires_explicit_consent(self):
        self.assertEqual(validate_marker_config(None), {"enabled": False, "confirmed_volunteer_demo": False, "allowed_codes": [], "min_reads": 3})
        for config in [{"enabled": True}, {"enabled": True, "allowed_codes": ["V01"]},
                       {"enabled": True, "confirmed_volunteer_demo": True}, {"method": "appearance"},
                       {"enabled": 1}, {"min_reads": True}, {"min_reads": 0}]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_marker_config(config)
        config = validate_marker_config({"enabled": True, "confirmed_volunteer_demo": True, "allowed_codes": ["V07", "V07"], "min_reads": 3})
        self.assertEqual(config["allowed_codes"], ["V07"])

    def test_printed_marker_has_quiet_zone_caption_and_real_decode(self):
        for code, size in [("V01", 224), ("V07", 800), ("V99", 320)]:
            with self.subTest(code=code, size=size):
                card = generate_qr_image(code, size)
                self.assertEqual(card.shape, (size, size, 3))
                self.assertEqual(card.dtype, np.uint8)
                self.assertTrue((card[:4] == 255).all())
                self.assertTrue((card[:, :4] == 255).all())
                self.assertTrue((card[int(size * .9):] < 128).any())
                decoded, points, _ = cv2.QRCodeDetector().detectAndDecode(card)
                self.assertEqual(decoded, f"TER:{code}")
                self.assertIsNotNone(points)
                found = MarkerReader([code]).read(card)
                self.assertEqual([item["code"] for item in found], [code])
                self.assertEqual(set(found[0]), {"code", "corners"})

    def test_real_multi_tag_frame_reads_two_authorized_codes(self):
        frame = np.full((300, 600, 3), 255, np.uint8)
        frame[25:265, 20:260] = generate_qr_image("V07", 240)
        frame[25:265, 330:570] = generate_qr_image("V08", 240)
        reader = MarkerReader(["V07", "V08"])
        self.assertEqual([item["code"] for item in reader.read(frame)], ["V07", "V08"])
        self.assertEqual(reader.duplicate_codes, [])

    def test_invalid_raw_payloads_and_unallowed_codes_are_not_returned(self):
        reader = MarkerReader(["V07"])
        payloads = ("https://example.com/private", "name:Jane Doe", "TER:V00", "TER:V07\n", "TER:V08", "TER:V07")
        reader._detector = Mock()
        reader._detector.detectAndDecodeMulti.return_value = (True, payloads, np.array([corners(10 + i * 40) for i in range(len(payloads))]), ())
        found = reader.read(np.zeros((100, 300, 3), np.uint8))
        self.assertEqual([item["code"] for item in found], ["V07"])
        self.assertNotIn("Jane", repr(found))
        self.assertNotIn("https", repr(found))
        reader._detector.detectAndDecode.assert_not_called()

    def test_empty_multi_result_uses_single_decode_fallback(self):
        reader = MarkerReader(["V07"])
        reader._detector = Mock()
        reader._detector.detectAndDecodeMulti.return_value = (False, (), None, ())
        reader._detector.detectAndDecode.return_value = ("TER:V07", np.array([corners()]), None)
        self.assertEqual(reader.read(np.zeros((100, 100), np.uint8))[0]["code"], "V07")
        reader._detector.detectAndDecode.assert_called_once()

    def test_duplicate_geometry_is_deduplicated_but_separate_tags_are_ambiguous(self):
        reader = MarkerReader(["V07"])
        reader._detector = Mock()
        first = np.array(corners(), dtype=np.float32)
        reader._detector.detectAndDecodeMulti.return_value = (True, ("TER:V07",) * 3,
                                                            np.array([first, np.roll(first + 1, 2, axis=0), corners(80)]), ())
        found = reader.read(np.zeros((150, 150, 3), np.uint8))
        self.assertEqual(len(found), 2)
        self.assertEqual(reader.duplicate_codes, ["V07"])
        reader._detector.detectAndDecodeMulti.return_value = (True, ("TER:V07",), np.array([first]), ())
        self.assertEqual(len(reader.read(np.zeros((150, 150, 3), np.uint8))), 1)
        self.assertEqual(reader.duplicate_codes, [])

    def test_invalid_geometry_is_ignored_and_decode_errors_do_not_leak_payloads(self):
        reader = MarkerReader(["V07"])
        reader._detector = Mock()
        reader._detector.detectAndDecodeMulti.return_value = (True, ("TER:V07",) * 3,
                                                            np.array([corners(-1), [[float("nan"), 0]] * 4, [[1, 1]] * 4]), ())
        self.assertEqual(reader.read(np.zeros((100, 100, 3), np.uint8)), [])
        reader._detector.detectAndDecodeMulti.side_effect = cv2.error("private QR data")
        reader._detector.detectAndDecode.side_effect = cv2.error("private QR data")
        with self.assertRaisesRegex(RuntimeError, "^QR decoding failed for this frame$"):
            reader.read(np.zeros((100, 100, 3), np.uint8))

    def test_minimum_reads_shared_codes_and_duplicate_exclusion(self):
        report = summarize_marker_cameras([
            summary("aerial", {"V01": 3, "V02": 2, "V03": 4, "V04": 4}),
            summary("ground", {"V01": 3, "V02": 8, "V03": 1, "V04": 4}, ["V04"]),
        ])
        self.assertEqual(report["shared_codes"], ["V01"])
        self.assertEqual(report["shared_count"], 1)
        self.assertEqual(report["method"], "explicit_qr_codes")
        self.assertEqual(report["scope"], "controlled_marker_demo")
        self.assertEqual(len(report["cameras"]["aerial"]["observations"]), 4)
        self.assertEqual(set(report["cameras"]["aerial"]), {"frames_processed", "frames_scanned", "read_failures", "observations", "duplicate_codes"})
        self.assertIn("ambiguous", report["warning"])

    def test_aggregate_strips_person_fields_raw_payloads_and_unsafe_evidence_paths(self):
        camera = summary("aerial", {"V07": 3})
        camera["person_id"] = "Jane Doe"
        camera["markers"]["observations"]["V07"].update(person_id="Jane", payload="TER:V07", evidence_file="../private.jpg")
        camera["markers"]["observations"]["https://private"] = {"read_frames": 3}
        report = summarize_marker_cameras([camera, summary("ground", {"V07": 3})])
        self.assertEqual(report["shared_count"], 1)
        evidence = report["cameras"]["aerial"]["observations"]["V07"]
        self.assertIsNone(evidence["evidence_file"])
        self.assertEqual(set(evidence), {"first_frame", "last_frame", "first_seconds", "last_seconds", "read_frames", "evidence_file"})
        self.assertNotIn("person_id", repr(report))
        self.assertNotIn("https://private", repr(report))
        self.assertNotIn("Jane", repr(report))

    def test_impossible_read_counts_and_missing_camera_never_inflate_shared_total(self):
        camera = summary("aerial", {"V07": 101})
        report = summarize_marker_cameras([camera, summary("ground", {"V07": 3})])
        self.assertEqual(report["shared_count"], 0)
        self.assertEqual(len(report["cameras"]["aerial"]["observations"]), 0)
        report = summarize_marker_cameras([summary("aerial", {"V07": 3})])
        self.assertEqual(report["shared_count"], 0)
        self.assertIn("Both camera", report["warning"])

    def test_shared_total_is_bounded_by_99_codes_and_distinct_camera_views(self):
        all_codes = {f"V{index:02d}": 3 for index in range(1, 100)}
        report = summarize_marker_cameras([summary("aerial", all_codes), summary("ground", all_codes)])
        self.assertEqual(report["shared_count"], 99)
        with self.assertRaises(ValueError):
            summarize_marker_cameras([summary("aerial", {"V07": 3})] * 2)
        with self.assertRaises(ValueError):
            generate_qr_image("V00")


if __name__ == "__main__":
    unittest.main()
