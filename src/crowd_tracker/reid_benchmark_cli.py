"""Evaluate a trained Re-ID checkpoint on a labelled manifest split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .reid_evaluation import LabeledEmbedding, evaluate_retrieval
from .reid_inference import TrainedReIdEncoder
from .reid_training import load_manifest


def _embeddings(samples, root: Path, encoder: TrainedReIdEncoder, batch_size: int) -> list[LabeledEmbedding]:
    result: list[LabeledEmbedding] = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        images = [cv2.imread(str(root / sample.path)) for sample in batch]
        if any(image is None for image in images):
            bad = next(sample for sample, image in zip(batch, images) if image is None)
            raise FileNotFoundError(f"Could not read Re-ID image: {root / bad.path}")
        for offset, (sample, embedding) in enumerate(zip(batch, encoder.embed_many_bgr(images)), start=1):
            result.append(LabeledEmbedding(str(start + offset), sample.person_id, sample.camera_id, embedding))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark an AG-ReID checkpoint with CMC and mAP.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="runs/reid/agreid_aerial_ground_metrics.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    root = Path(args.root)
    encoder = TrainedReIdEncoder(args.checkpoint, args.device, args.image_size)
    if args.batch_size < 1:
        raise ValueError("batch-size must be >= 1")
    queries = _embeddings(load_manifest(args.manifest, "test_query_aerial"), root, encoder, args.batch_size)
    gallery = _embeddings(load_manifest(args.manifest, "test_gallery_ground"), root, encoder, args.batch_size)
    result = evaluate_retrieval(queries, gallery)
    payload = {
        "dataset": "AG-ReID v1",
        "protocol": "C0 aerial query to C3 ground gallery (official held-out split)",
        "checkpoint": str(args.checkpoint),
        "method": encoder.method_label,
        **result,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
