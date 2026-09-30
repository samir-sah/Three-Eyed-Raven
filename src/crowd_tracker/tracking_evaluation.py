"""Reproducible single-camera detection and tracking evaluation utilities.

Inputs use the same JSONL shape as ``observations.jsonl``: one object per person
with ``frame_index``, ``track_id`` and ``bbox_xyxy``. Ground-truth IDs must be
stable within the evaluated camera sequence.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class LabeledBox:
    frame_index: int
    track_id: str
    bbox_xyxy: tuple[float, float, float, float]


def _maximum_weight_track_assignment(pair_counts: dict[tuple[str, str], int]) -> int:
    """Return the global one-to-one identity assignment score.

    IDF1 uses a sequence-wide matching between ground-truth and predicted track
    identities, unlike frame-wise detection association. This small Hungarian
    implementation keeps the metric reproducible without an optional SciPy
    dependency.
    """
    if not pair_counts:
        return 0
    ground_truth_ids = sorted({ground_truth_id for ground_truth_id, _ in pair_counts})
    prediction_ids = sorted({prediction_id for _, prediction_id in pair_counts})
    size = max(len(ground_truth_ids), len(prediction_ids))
    maximum = max(pair_counts.values())
    weights = [[0] * size for _ in range(size)]
    gt_index = {track_id: index for index, track_id in enumerate(ground_truth_ids)}
    prediction_index = {track_id: index for index, track_id in enumerate(prediction_ids)}
    for (ground_truth_id, prediction_id), count in pair_counts.items():
        weights[gt_index[ground_truth_id]][prediction_index[prediction_id]] = count

    # Hungarian algorithm for minimum cost; negate the objective via max-weight.
    costs = [[maximum - weight for weight in row] for row in weights]
    u = [0] * (size + 1)
    v = [0] * (size + 1)
    matching = [0] * (size + 1)
    path = [0] * (size + 1)
    for row in range(1, size + 1):
        matching[0] = row
        column0 = 0
        min_values = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column0] = True
            row0 = matching[column0]
            delta = float("inf")
            next_column = 0
            for column in range(1, size + 1):
                if used[column]:
                    continue
                current = costs[row0 - 1][column - 1] - u[row0] - v[column]
                if current < min_values[column]:
                    min_values[column] = current
                    path[column] = column0
                if min_values[column] < delta:
                    delta = min_values[column]
                    next_column = column
            for column in range(size + 1):
                if used[column]:
                    u[matching[column]] += delta
                    v[column] -= delta
                else:
                    min_values[column] -= delta
            column0 = next_column
            if matching[column0] == 0:
                break
        while True:
            previous_column = path[column0]
            matching[column0] = matching[previous_column]
            column0 = previous_column
            if column0 == 0:
                break

    return sum(weights[row - 1][column - 1] for column, row in enumerate(matching[1:], start=1) if row)


def iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    """Return intersection-over-union for two XYXY boxes."""
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def evaluate_tracking(
    ground_truth: list[LabeledBox],
    predictions: list[LabeledBox],
    iou_threshold: float = 0.5,
) -> dict:
    """Compute detection precision/recall and clear MOT-style tracking metrics.

    Association uses one-to-one maximum-IoU matching per frame. ``id_switches``
    counts a change in matched predicted local ID for a ground-truth identity
    between consecutive matched observations.
    """
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1].")

    gt_by_frame: dict[int, list[LabeledBox]] = defaultdict(list)
    prediction_by_frame: dict[int, list[LabeledBox]] = defaultdict(list)
    for box in ground_truth:
        gt_by_frame[box.frame_index].append(box)
    for box in predictions:
        prediction_by_frame[box.frame_index].append(box)

    matches = false_positives = false_negatives = id_switches = 0
    matched_ious: list[float] = []
    previous_match: dict[str, str] = {}
    identity_pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    for frame_index in sorted(set(gt_by_frame) | set(prediction_by_frame)):
        frame_gt = gt_by_frame[frame_index]
        frame_predictions = prediction_by_frame[frame_index]
        pairs = [
            (iou(gt.bbox_xyxy, prediction.bbox_xyxy), gt_index, prediction_index)
            for gt_index, gt in enumerate(frame_gt)
            for prediction_index, prediction in enumerate(frame_predictions)
        ]
        used_gt: set[int] = set()
        used_predictions: set[int] = set()
        for overlap, gt_index, prediction_index in sorted(pairs, reverse=True):
            if overlap < iou_threshold or gt_index in used_gt or prediction_index in used_predictions:
                continue
            used_gt.add(gt_index)
            used_predictions.add(prediction_index)
            matches += 1
            matched_ious.append(overlap)
            gt_id = frame_gt[gt_index].track_id
            prediction_id = frame_predictions[prediction_index].track_id
            identity_pair_counts[(gt_id, prediction_id)] += 1
            if gt_id in previous_match and previous_match[gt_id] != prediction_id:
                id_switches += 1
            previous_match[gt_id] = prediction_id
        false_negatives += len(frame_gt) - len(used_gt)
        false_positives += len(frame_predictions) - len(used_predictions)

    total_ground_truth = len(ground_truth)
    total_predictions = len(predictions)
    precision = matches / total_predictions if total_predictions else 0.0
    recall = matches / total_ground_truth if total_ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    id_true_positives = _maximum_weight_track_assignment(identity_pair_counts)
    id_false_positives = total_predictions - id_true_positives
    id_false_negatives = total_ground_truth - id_true_positives
    id_precision = id_true_positives / (id_true_positives + id_false_positives) if id_true_positives + id_false_positives else 0.0
    id_recall = id_true_positives / (id_true_positives + id_false_negatives) if id_true_positives + id_false_negatives else 0.0
    idf1 = 2 * id_precision * id_recall / (id_precision + id_recall) if id_precision + id_recall else 0.0
    return {
        "iou_threshold": iou_threshold,
        "ground_truth_boxes": total_ground_truth,
        "predicted_boxes": total_predictions,
        "matches": matches,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "id_switches": id_switches,
        "id_true_positives": id_true_positives,
        "id_false_positives": id_false_positives,
        "id_false_negatives": id_false_negatives,
        "id_precision": round(id_precision, 4),
        "id_recall": round(id_recall, 4),
        "idf1": round(idf1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mota": round(1 - (false_positives + false_negatives + id_switches) / total_ground_truth, 4) if total_ground_truth else None,
        "motp": round(sum(matched_ious) / len(matched_ious), 4) if matched_ious else None,
    }


def parse_labeled_box(raw: dict, source_name: str) -> LabeledBox:
    try:
        frame_index = int(raw["frame_index"])
        track_id = str(raw["track_id"])
        coords = raw["bbox_xyxy"]
        if len(coords) != 4:
            raise ValueError
        bbox = tuple(float(value) for value in coords)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid {source_name} record; expected frame_index, track_id and four bbox_xyxy values.") from error
    if frame_index < 0 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError(f"Invalid {source_name} record geometry.")
    return LabeledBox(frame_index=frame_index, track_id=track_id, bbox_xyxy=bbox)
