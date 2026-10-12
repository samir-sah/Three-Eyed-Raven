# Extended crowd detection experiment

This workflow trains image-level person detection and evaluates aggregate crowd counts. It does not run identity matching or cross-camera association, and it never replaces the dashboard's current models automatically.

## Run

From the repository root, using its CUDA-enabled virtual environment:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -u -m crowd_tracker.extended_detection_cli --hours 5
```

Each invocation needs a new experiment name/directory. `--smoke` uses one epoch and 0.5% of training images to test checkpointing/evaluation; it does not become the active dashboard experiment. Extended runs use every training image.

The run audits every person label, records the annotation digest and filename overlap check, and uses an absolute dataset YAML to avoid Ultralytics' relative dataset-root ambiguity.

## Schedule

1. Continue from the existing 50-epoch VisDrone checkpoint at 640 pixels (60% of the training budget).
2. Fine-tune the resulting checkpoint at 960 pixels (40% of the budget), using a smaller batch for the 4 GB GPU.
3. Compare the old model at both resolutions with both new candidates. Measure detection mAP and count MAE/RMSE. Fit confidence thresholds on a sequence-grouped calibration partition and report counting error on a separate partition.

All training uses the full 6,471-image split; validation contains 548 images. Validation also selects best checkpoints, so the final comparisons are development results, not independent test scores. More training is not guaranteed to improve validation performance. Early stopping is enabled, so a plateau can finish a stage sooner.

## Progress and recovery

Open `/training` in the local dashboard. It refreshes every ten seconds and shows saved status, stage, epoch, batches, checkpoint paths, failures, and final comparisons.

Artifacts live under `runs/experiments/<name>/`:

- `status.json`: atomic progress snapshots (about every 15 seconds while training).
- `dataset_audit.json`: dataset counts and annotation digest.
- `visdrone_person_640/weights/` and `visdrone_person_960/weights/`: best/latest checkpoint each epoch and periodic checkpoints every ten epochs.
- `*_evaluation.json`: per-model count and detection metrics.
- `comparison.json` and `REPORT.md`: final comparison, including stage errors.

The worker requests temporary Windows idle-sleep inhibition and releases it at exit. Keep the laptop plugged in with its lid open; shutdown, forced sleep, or lid-close policy can still interrupt it. Checkpoints preserve completed epochs, not an unfinished batch.

An interrupted training checkpoint can be resumed using the existing detector CLI:

```powershell
.\.venv\Scripts\python.exe -m crowd_tracker.train_detector_cli --data runs\experiments\<name>\dataset.yaml --resume runs\experiments\<name>\visdrone_person_640\weights\last.pt --workers 2
```

That resumes the selected training stage only. The experiment controller will not resume automatically; review the saved stage state before scheduling subsequent evaluation. The five-hour budget applies to training, with additional time for preparation, validation, and final analysis.
