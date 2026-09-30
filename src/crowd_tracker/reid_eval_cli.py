"""Evaluate exported embeddings using a simple labeled JSON manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .reid_evaluation import LabeledEmbedding, evaluate_retrieval


def _load(items: list[dict]) -> list[LabeledEmbedding]:
    return [
        LabeledEmbedding(
            sample_id=item["sample_id"],
            person_id=str(item["person_id"]),
            camera_id=item["camera_id"],
            embedding=np.asarray(item["embedding"], dtype=np.float32) / np.linalg.norm(item["embedding"]),
        )
        for item in items
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate labeled Re-ID query/gallery embeddings.")
    parser.add_argument("--manifest", required=True, help="JSON with queries and gallery embedding arrays.")
    parser.add_argument("--output", default=None, help="Optional metrics JSON destination.")
    args = parser.parse_args()
    payload = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    result = evaluate_retrieval(_load(payload["queries"]), _load(payload["gallery"]))
    rendered = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
