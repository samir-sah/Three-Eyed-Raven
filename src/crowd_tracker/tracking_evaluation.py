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
    return {
        "iou_threshold": iou_threshold,
        "ground_truth_boxes": total_ground_truth,
        "predicted_boxes": total_predictions,
        "matches": matches,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "id_switches": id_switches,
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
