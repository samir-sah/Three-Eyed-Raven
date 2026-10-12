"""Create a reproducible aerial-to-ground AG-ReID manifest.

AG-ReID v1 uses C0/C1/C2 for UAV imagery and C3 for CCTV imagery.  The
official training directory is kept separate from the query/gallery folders,
so evaluation always uses identities and images withheld by the dataset split.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


_NAME = re.compile(r"^P(?P<person>\d+).*?C(?P<camera>[0-3])F", re.IGNORECASE)


def _record(path: Path, root: Path, split: str) -> dict | None:
    match = _NAME.match(path.name)
    if match is None:
        return None
    return {
        "path": path.relative_to(root).as_posix(),
        "person_id": match.group("person"),
        "camera_id": f"c{match.group('camera')}",
        "split": split,
    }


def _scan(root: Path, directory: str, split: str) -> list[dict]:
    location = root / directory
    if not location.is_dir():
        raise FileNotFoundError(f"AG-ReID directory is missing: {location}")
    records = [record for path in sorted(location.glob("*.jpg")) if (record := _record(path, root, split))]
    if not records:
        raise ValueError(f"No valid AG-ReID images found in {location}.")
    return records


def build_manifest(dataset_root: str | Path) -> tuple[list[dict], dict]:
    root = Path(dataset_root).resolve()
    train = _scan(root, "bounding_box_train", "train")
    # Evaluation direction reflects the dashboard use case: an aerial track is
    # searched against a ground-surveillance gallery.
    queries = _scan(root, "query_all_c0", "test_query_aerial")
    gallery = _scan(root, "bounding_box_test_all_c3", "test_gallery_ground")
    records = train + queries + gallery
    counts = Counter(record["split"] for record in records)
    metadata = {
        "dataset": "AG-ReID v1",
        "training_images": counts["train"],
        "training_identities": len({record["person_id"] for record in train}),
        "test_aerial_queries": counts["test_query_aerial"],
        "test_ground_gallery": counts["test_gallery_ground"],
        "evaluation": "C0 aerial query to C3 ground gallery; official held-out folders.",
    }
    return records, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an AG-ReID aerial-to-ground JSONL manifest.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", default="data/processed/agreid_aerial_ground_manifest.jsonl")
    args = parser.parse_args()
    records, metadata = build_manifest(args.dataset_root)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    destination.with_suffix(".summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(destination), **metadata}, indent=2))


if __name__ == "__main__":
    main()
