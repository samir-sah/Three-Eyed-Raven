# Detection target follow-up: tiling and a larger model

Scope: anonymous image-level person detection and aggregate crowd counts. No
identity embeddings, cross-camera linking, or automatic model deployment.

## Completed tiling comparison

`runs/experiments/tiled_full_20261012/comparison.json` evaluates all 548 VisDrone
validation images with the previously trained YOLO11n checkpoint.

| Configuration | mAP50 | Count MAE | Count RMSE | Evaluation images/s |
| --- | ---: | ---: | ---: | ---: |
| Whole-image control | 61.50% | 7.82 | 15.80 | 28.99 |
| 640px tiles + whole image | 64.15% | 7.22 | 13.79 | 6.79 |
| 960px tiles + whole image | 63.39% | 7.69 | 15.50 | 14.26 |

This uses Ultralytics matching and AP with original-image predictions. Its
predict-mode preprocessing differs from `model.val`, so use the whole-image
control here for the matched comparison, not an earlier validation number.
Whole images are inferred at 1536px, tiles at 960px, with 20% overlap. Internal
border-truncated detections are filtered and global merge NMS uses IoU 0.50.
Confidence is 0.001 for AP and maximum detections are 1000 per original image.
Counts use the existing 125-image calibration / 423-image report partition.
Throughput includes image IO, all tile passes, merging, and metrics; it is not
live-camera FPS. The 70% target was not reached.

Of 13,969 ground-truth boxes, 12,377 have original-image area below 32 squared
pixels. At confidence 0.20 and IoU 0.50, whole-image predictions cover 7,467 of
these small boxes; 640px tiles cover 8,155. This diagnostic allows any matching
prediction and is **not** one-to-one recall or identity matching. The per-setting
JSON files record size coverage and the 30 images with the most uncovered boxes.

Reproduce the tiled experiment with a new name:

```powershell
cd S:\three_eyed_raven
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m crowd_tracker.tiled_detection_cli --name tiled_new_run --checkpoint runs\experiments\extended_detection_20261012\visdrone_person_960\weights\best.pt
```

## Larger-model training

The separately named `larger_detection_20261012_w0` experiment trains YOLO11s from
official COCO-pretrained weights on all 6,471 VisDrone training images. Its
approximately three-hour training budget, 960px input, batch 1, AMP, and zero data-loader child processes
were preceded by a successful one-epoch infrastructure smoke test. The smoke
uses 0.5% of the training split and is not model-quality evidence.

YOLO11s has approximately 9.46 million pretrained parameters; the previous nano
detector has approximately 2.6 million. More capacity and training are hypotheses,
not guarantees of improvement. A three-hour exploratory run need not converge.

```powershell
cd S:\three_eyed_raven
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -u -m crowd_tracker.larger_detection_cli --name larger_new_run --hours 3
```

The root-level `yolo11s.pt` must already exist. The official source used was
https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.pt;
the run records its SHA-256. Dataset labels are audited and hashed. No dataset
images or checkpoints are committed to Git.

After training, the worker compares the previous nano model and the candidate
at 960px and 1536px, and tests candidate augmentation at 1536px. All evaluations
use the same full validation split. Results include mAP50, mAP50-95 and calibrated
count errors, including regressions. Existing models remain unchanged.

Progress: <http://127.0.0.1:3000/training>, or the named experiment's `status.json`.
Logs: `runs/experiments/larger_detection_20261012_w0.stdout.log` and `.stderr.log`.
Final results: `comparison.json` and `REPORT.md` in its experiment directory.
Keep the laptop plugged in and lid open. The worker temporarily inhibits idle
sleep, but cannot prevent shutdown or a forced/lid-triggered sleep.

Checkpoints save after completed epochs. Do not run duplicate GPU jobs. If the
process fails, preserve logs and weights and diagnose before resuming. Future
invocations require a fresh name to avoid overwriting experiments.

All reported scores are development validation results. Validation selects
checkpoints and settings; independent, scene-disjoint testing remains necessary.

### Memory recovery

The original `larger_detection_20261012` run failed before completing epoch 1:
an OpenCV image-loading child process could not allocate 4.41 MB of system RAM.
This was not a CUDA out-of-memory error. Its logs remain preserved. A concurrent
dashboard build may have contributed to memory pressure; that is not established
as the sole cause. Two stale test-process entries from the morning were checked;
the surviving owned test launcher was stopped. User apps were not closed.

The retry uses `--workers 0` for training and evaluation, avoiding RAM-heavy
Windows data-loader children. It starts from the original official weights
because the failed run had no saved training checkpoint. Its budget is 2.95 hours.
The worker records this change and keeps the same dataset, model and scoring.
Do not run builds or duplicate training jobs alongside this experiment.
