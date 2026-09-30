"""Standard retrieval metrics for a labeled person Re-ID benchmark split."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LabeledEmbedding:
    sample_id: str
    person_id: str
    camera_id: str
    embedding: np.ndarray


def evaluate_retrieval(queries: list[LabeledEmbedding], gallery: list[LabeledEmbedding]) -> dict:
    """Return CMC Rank-1/5/10 and mAP, excluding same-camera same-person images."""
    if not queries or not gallery:
        raise ValueError("queries and gallery must both contain at least one labeled embedding.")
    valid_queries = 0
    average_precisions: list[float] = []
    rank_hits = {1: 0, 5: 0, 10: 0}

    for query in queries:
        ranked = sorted(
            (
                (float(np.dot(query.embedding, item.embedding)), item)
                for item in gallery
                if not (item.person_id == query.person_id and item.camera_id == query.camera_id)
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        matches = [item.person_id == query.person_id for _, item in ranked]
        if not any(matches):
            continue
        valid_queries += 1
        correct = 0
        precision_sum = 0.0
        for rank, is_match in enumerate(matches, start=1):
            if is_match:
                correct += 1
                precision_sum += correct / rank
        average_precisions.append(precision_sum / correct)
        for cutoff in rank_hits:
            rank_hits[cutoff] += int(any(matches[:cutoff]))

    if not valid_queries:
        raise ValueError("No query identity appears in a different-camera gallery image.")
    return {
        "queries": len(queries),
        "valid_queries": valid_queries,
        "rank_1": round(rank_hits[1] / valid_queries, 4),
        "rank_5": round(rank_hits[5] / valid_queries, 4),
        "rank_10": round(rank_hits[10] / valid_queries, 4),
        "mAP": round(float(np.mean(average_precisions)), 4),
    }
