"""CPU-only QR session tests: anonymous codes, frame counts and tag-only files."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from crowd_tracker.marker_codes import summarize_marker_cameras
from crowd_tracker.marker_session import MarkerSession


def marker(code="V07", x=100, y=70, size=80):
    return {"code": code, "corners": [[x, y], [x + size, y], [x + size, y + size], [x, y + size]]}


def frame_fixture():
    # A colored synthetic tag on a contrasting scene makes accidental inclusion
    # of anything outside the marker quadrilateral easy to detect.
    frame = np.full((240, 360, 3), (0, 0, 255), dtype=np.uint8)
    frame[70:151, 100:181] = (0, 255, 0)
    return frame


def scripted_reader(script):
    reader = Mock()
    reader.duplicate_codes = []
    remaining = iter(script)

    def read(_frame):
        result = next(remaining)
        if isinstance(result, Exception):
            raise result
        found, duplicates = result
        reader.duplicate_codes = list(duplicates)
        return found

    reader.read.side_effect = read
    return reader


def read_log(root):
    return [json.loads(line) for line in (root / "markers.jsonl").read_text(encoding="utf-8").splitlines()]


class MarkerSessionTests(unittest.TestCase):
    def test_distinct_frame_reads_are_not_inflated_by_duplicate_locations(self):
        reader = scripted_reader([
            ([marker()], []),
            ([marker(), marker(x=200)], ["V07"]),
            ([], []),
            ([marker()], []),
        ])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch("crowd_tracker.marker_session.MarkerReader", return_value=reader), \
                 patch("cv2.imwrite", wraps=cv2.imwrite) as save:
                session = MarkerSession(root, ["V07"])
                try:
                    for index in range(4):
                        session.read(frame_fixture(), index, index / 10)
                    result = session.summary(4)
                finally:
                    session.close()
            observation = result["observations"]["V07"]
            self.assertEqual(observation["read_frames"], 3)
            self.assertEqual((observation["first_frame"], observation["last_frame"]), (0, 3))
            self.assertEqual((observation["first_seconds"], observation["last_seconds"]), (0, .3))
            self.assertEqual(result["frames_scanned"], 4)
            self.assertEqual(result["frames_processed"], 4)
            self.assertEqual(result["read_failures"], 0)
            self.assertEqual(result["duplicate_codes"], ["V07"])
            save.assert_called_once()
            rows = read_log(root)
            self.assertEqual([row["frame_index"] for row in rows], [0, 1, 2, 3])
            self.assertEqual([len(row["markers"]) for row in rows], [1, 2, 0, 1])
            self.assertEqual(rows[1]["duplicate_codes"], ["V07"])
            self.assertEqual(rows[2]["markers"], [])
            self.assertTrue(all(set(row) == {"frame_index", "timestamp_seconds", "read_failed", "markers", "duplicate_codes"} for row in rows))

    def test_evidence_is_256_pixels_and_contains_only_decoded_tag_quadrilateral(self):
        reader = scripted_reader([([marker()], [])])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch("crowd_tracker.marker_session.MarkerReader", return_value=reader), \
                 patch("cv2.getPerspectiveTransform", wraps=cv2.getPerspectiveTransform) as transform:
                session = MarkerSession(root, ["V07"])
                try:
                    session.read(frame_fixture(), 0, 0)
                    result = session.summary(1)
                finally:
                    session.close()
            evidence = cv2.imread(str(root / "marker_evidence" / "V07.jpg"))
            self.assertEqual(evidence.shape, (256, 256, 3))
            self.assertTrue((evidence[:, :, 1] > 250).all())
            self.assertTrue((evidence[:, :, 2] < 5).all())
            np.testing.assert_array_equal(transform.call_args.args[0], np.asarray(marker()["corners"], dtype=np.float32))
            np.testing.assert_array_equal(transform.call_args.args[1], np.asarray([[0, 0], [255, 0], [255, 255], [0, 255]], dtype=np.float32))
            self.assertEqual(result["observations"]["V07"]["evidence_file"], "V07.jpg")
            self.assertEqual(sorted(file.name for file in root.iterdir()), ["marker_evidence", "markers.jsonl"])
            self.assertEqual(set(read_log(root)[0]["markers"][0]), {"code", "corners"})

    def test_duplicate_only_reads_have_null_evidence_and_no_image_written(self):
        reader = scripted_reader([([marker(), marker(x=200)], ["V07"])] * 2)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch("crowd_tracker.marker_session.MarkerReader", return_value=reader), patch("cv2.imwrite") as save:
                session = MarkerSession(root, ["V07"])
                try:
                    session.read(frame_fixture(), 0, 0)
                    session.read(frame_fixture(), 1, .1)
                    result = session.summary(2)
                finally:
                    session.close()
            self.assertEqual(result["observations"]["V07"]["read_frames"], 2)
            self.assertIsNone(result["observations"]["V07"]["evidence_file"])
            self.assertEqual(list((root / "marker_evidence").iterdir()), [])
            self.assertEqual(result["duplicate_codes"], ["V07"])
            save.assert_not_called()

    def test_decoder_failures_are_logged_without_foreign_error_contents(self):
        private_text = "https://example.invalid/private?name=Jane"
        reader = scripted_reader([RuntimeError(private_text), ValueError(private_text), cv2.error(private_text), ([], [])])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch("crowd_tracker.marker_session.MarkerReader", return_value=reader):
                session = MarkerSession(root, ["V07"])
                try:
                    for index in range(4):
                        self.assertEqual(session.read(frame_fixture(), index, index / 10), [])
                    result = session.summary(4)
                finally:
                    session.close()
            self.assertEqual(result["frames_scanned"], 4)
            self.assertEqual(result["read_failures"], 3)
            self.assertEqual(result["observations"], {})
            self.assertEqual([row["read_failed"] for row in read_log(root)], [True, True, True, False])
            contents = (root / "markers.jsonl").read_text(encoding="utf-8")
            self.assertNotIn(private_text, contents)
            self.assertNotIn("Jane", contents)
            self.assertTrue(all(row["markers"] == [] for row in read_log(root)))

    def test_mocked_decoder_foreign_payloads_are_removed_before_session_storage(self):
        # Compose the real strict reader with a mocked decoder, so the session's
        # saved log is checked at the full privacy boundary rather than merely
        # assuming the reader filtered unauthorized contents.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            session = MarkerSession(root, ["V07"])
            session.reader._detector = Mock()
            session.reader._detector.detectAndDecodeMulti.return_value = (
                True, ("https://private.invalid", "TER:V08", "TER:V07"),
                np.asarray([marker()["corners"]] * 3, dtype=np.float32), (),
            )
            try:
                found = session.read(frame_fixture(), 0, 0)
                result = session.summary(1)
            finally:
                session.close()
            self.assertEqual([tag["code"] for tag in found], ["V07"])
            self.assertEqual(list(result["observations"]), ["V07"])
            contents = (root / "markers.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("https://", contents)
            self.assertNotIn("TER:", contents)
            self.assertNotIn("V08", contents)

    def test_later_clear_read_can_supply_evidence_but_does_not_erase_ambiguity(self):
        reader = scripted_reader([([marker(), marker(x=200)], ["V07"]), ([marker()], [])])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch("crowd_tracker.marker_session.MarkerReader", return_value=reader):
                session = MarkerSession(root, ["V07"])
                try:
                    session.read(frame_fixture(), 0, 0)
                    self.assertIsNone(session.observations["V07"]["evidence_file"])
                    session.read(frame_fixture(), 1, .1)
                    result = session.summary(2)
                finally:
                    session.close()
            self.assertEqual(result["observations"]["V07"]["evidence_file"], "V07.jpg")
            self.assertEqual(result["duplicate_codes"], ["V07"])
            other = {**result, "duplicate_codes": []}
            shared = summarize_marker_cameras([{"camera_id": "aerial", "markers": result}, {"camera_id": "ground", "markers": other}], 2)
            self.assertEqual(shared["shared_count"], 0)

    def test_close_flushes_log_and_is_idempotent(self):
        reader = scripted_reader([([], [])])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch("crowd_tracker.marker_session.MarkerReader", return_value=reader):
                session = MarkerSession(root, ["V07"])
                try:
                    session.read(frame_fixture(), 0, 0)
                finally:
                    session.close()
            self.assertTrue(session.log.closed)
            session.close()
            self.assertEqual(len(read_log(root)), 1)

    def test_repeated_or_invalid_frame_indices_do_not_inflate_reads_or_logs(self):
        reader = scripted_reader([([marker()], []), ([marker()], [])])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch("crowd_tracker.marker_session.MarkerReader", return_value=reader):
                session = MarkerSession(root, ["V07"])
                try:
                    session.read(frame_fixture(), 0, 0)
                    for index in (0, -1, True, False, .5, "1", None):
                        with self.subTest(frame_index=index), self.assertRaises(ValueError):
                            session.read(frame_fixture(), index, .1)
                    self.assertEqual(session.frames, 1)
                    self.assertEqual(session.failures, 0)
                    self.assertEqual(session.observations["V07"]["read_frames"], 1)
                    reader.read.assert_called_once()
                    session.read(frame_fixture(), 1, .1)
                    result = session.summary(2)
                finally:
                    session.close()
            self.assertEqual(result["observations"]["V07"]["read_frames"], 2)
            self.assertEqual([row["frame_index"] for row in read_log(root)], [0, 1])

    def test_invalid_timestamps_are_rejected_before_scan_or_log_changes(self):
        reader = scripted_reader([([], [])])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch("crowd_tracker.marker_session.MarkerReader", return_value=reader):
                session = MarkerSession(root, ["V07"])
                try:
                    for seconds in (-1, float("nan"), float("inf"), float("-inf"), True, False, "0", None):
                        with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                            session.read(frame_fixture(), 0, seconds)
                    self.assertEqual(session.frames, 0)
                    self.assertEqual(session.failures, 0)
                    reader.read.assert_not_called()
                    session.read(frame_fixture(), 0, 0)
                finally:
                    session.close()
            self.assertEqual(len(read_log(root)), 1)
            self.assertEqual(read_log(root)[0]["timestamp_seconds"], 0)

    def test_evidence_write_failure_is_explicit_and_close_still_releases_log(self):
        reader = scripted_reader([([marker()], [])])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch("crowd_tracker.marker_session.MarkerReader", return_value=reader), patch("cv2.imwrite", return_value=False):
                session = MarkerSession(root, ["V07"])
                try:
                    with self.assertRaisesRegex(RuntimeError, "preserve decoded QR tag evidence"):
                        session.read(frame_fixture(), 0, 0)
                    self.assertIsNone(session.observations["V07"]["evidence_file"])
                finally:
                    session.close()
            self.assertTrue(session.log.closed)


if __name__ == "__main__":
    unittest.main()
