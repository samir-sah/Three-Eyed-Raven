import tempfile
import unittest
from pathlib import Path

from crowd_tracker.reid_training import TripletManifestDataset, load_manifest


class ReIdTrainingTests(unittest.TestCase):
    def test_manifest_filters_by_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "manifest.jsonl"
            manifest.write_text(
                '{"path":"a.jpg","person_id":"1","camera_id":"a","split":"train"}\n'
                '{"path":"b.jpg","person_id":"1","camera_id":"b","split":"test"}\n',
                encoding="utf-8",
            )
            self.assertEqual(len(load_manifest(manifest, "train")), 1)

    def test_triplet_dataset_requires_repeat_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "manifest.jsonl"
            manifest.write_text(
                '{"path":"a.jpg","person_id":"1","camera_id":"a","split":"train"}\n'
                '{"path":"b.jpg","person_id":"2","camera_id":"b","split":"train"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Triplet"):
                TripletManifestDataset(load_manifest(manifest))
