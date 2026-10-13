# Ground detection baseline — 13 October 2026

The current upload detector (COCO YOLO11n) was the strongest of the four existing checkpoints in this sampled ground-camera audit. Keep it as the ground baseline. The extended aerial nano model remains the stronger tested aerial candidate. Neither this audit nor the preceding YOLO11s experiment automatically changes upload settings.

## Measured comparison

Corrected experiment: `runs/experiments/ground_detection_20261013_corrected`. The preliminary run without MOT coordinate-origin conversion is preserved but is not used on the dashboard.

Sampled 215 images, approximately one per second, from seven MOT17 public training scenes. Four scenes (137 images) calibrate thresholds; three separate scenes (78 images, 965 pedestrian annotations) supply the report below. All models use 640px, person class 0, NMS IoU .5 and maximum 300 predictions. No tracker, person embeddings or cross-camera matching is called.

| Detector | Report mAP50 | Recall at .35 | Precision at .35 | Count MAE at .35 |
| --- | --- | --- | --- | --- |
| COCO YOLO11n (website upload model) | 57.40% | 43.21% | 91.05% | 6.55 |
| Historical VisDrone 50-epoch model | 27.05% | 12.95% | 82.78% | 10.46 |
| Extended aerial YOLO11n | 35.13% | 14.61% | 87.04% | 10.29 |
| Exploratory aerial YOLO11s | 22.66% | 7.56% | 93.59% | 11.37 |

At a .05 threshold selected by calibration F1, COCO YOLO11n report recall was 58.65%, precision 59.02% and count MAE 2.95. At .35, it had 417 true positives and 41 false positives; at .05, 566 true positives and 393 false positives. The improved count error is partly offsetting missed people with false detections. This is not sufficient evidence to ship .05 as the upload confidence threshold.

## What this establishes

- The old 17.12% tracking recall is not the current upload detector's detection performance. Its score also used different frames, filtering and the tracker, so it is not directly comparable to this table.
- Aerial fine-tuning does not give the strongest ground detector in this comparison. Training-domain differences are consistent with the result, but model version, training history and other factors also differ.
- The COCO checkpoint is a useful ground baseline. Ground-specific data and careful confidence calibration are better justified next steps than deploying the unsuccessful aerial small checkpoint.
- A count can be numerically close while individual detections are wrong. Report detection quality alongside count error.

## Limits

This is a custom detection protocol documented in `DETECTION_EVALUATION_PROTOCOL.md`, not an official MOTChallenge leaderboard result. Counts include marked class-1 pedestrians at every annotated visibility level, using full-body annotations, and ignore supplied non-target/unmarked boxes. The target category is not identical to every visible person in arbitrary footage.

MOT17 was already used for a historical project benchmark. The report scenes are separate from calibration but are not a newly untouched final test. With only three report scenes, no reliable broad generalization or 70% accuracy claim is made. No deployment changed.

Inspect the new **Model comparisons** sidebar item or supporting tab. It separates the fresh detector audit from the collapsed historical tracking report, and lets you compare the original .35 confidence against calibrated thresholds.

Next: prepare permitted representative aerial and ground clips, agree on labels, reserve whole scenes for an independent final test, and then choose ground training experiments based on these measured failure cases.
