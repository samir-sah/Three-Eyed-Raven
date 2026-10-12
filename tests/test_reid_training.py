import tempfile
import unittest
from pathlib import Path

from crowd_tracker.reid_training import TripletManifestDataset, balanced_cross_camera_subset, load_manifest
from crowd_tracker.agreid_prepare_cli import build_manifest


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

    def test_agreid_manifest_keeps_train_and_held_out_protocol_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for folder, filenames in {
                "bounding_box_train": ["P0001T00001A0C0F1.jpg", "P0001T00001A0C3F1.jpg"],
                "query_all_c0": ["P0100T00001A0C0F1.jpg"],
                "bounding_box_test_all_c3": ["P0100T00001A0C3F1.jpg"],
            }.items():
                directory = root / folder
                directory.mkdir()
                for name in filenames:
                    (directory / name).touch()
            records, metadata = build_manifest(root)
            self.assertEqual(metadata["training_images"], 2)
            self.assertEqual({record["split"] for record in records}, {"train", "test_query_aerial", "test_gallery_ground"})

    def test_balanced_subset_preserves_two_cameras_per_identity(self):
        samples = [
            *[load_manifest_sample("1", "c0", index) for index in range(4)],
            *[load_manifest_sample("1", "c3", index) for index in range(4)],
            *[load_manifest_sample("2", "c0", index) for index in range(4)],
            *[load_manifest_sample("2", "c3", index) for index in range(4)],
        ]
        subset = balanced_cross_camera_subset(samples, 6)
        self.assertEqual(len(subset), 6)
        self.assertEqual({sample.camera_id for sample in subset if sample.person_id == "1"}, {"c0", "c3"})
        self.assertEqual({sample.camera_id for sample in subset if sample.person_id == "2"}, {"c0", "c3"})


def load_manifest_sample(person_id, camera_id, index):
    from crowd_tracker.reid_training import ReIdSample
    return ReIdSample(f"{person_id}_{camera_id}_{index}.jpg", person_id, camera_id, "train")
