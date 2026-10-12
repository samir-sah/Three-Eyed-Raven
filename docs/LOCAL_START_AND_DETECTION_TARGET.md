# Local startup and the detection target

## Start the local website

The existing environment and dashboard dependencies are already installed on this machine.
Open two **PowerShell** terminals and run each block separately.

Terminal 1 — read-only run API:

```powershell
cd S:\three_eyed_raven
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m uvicorn crowd_tracker.api:app --host 127.0.0.1 --port 8000
```

Terminal 2 — dashboard:

```powershell
cd S:\three_eyed_raven\dashboard
npm.cmd run dev -- --hostname 127.0.0.1
```

Open <http://127.0.0.1:3000>. Completed extended-training results are at
<http://127.0.0.1:3000/training>. Leave both terminals open; Ctrl+C stops each service.
Starting the website does not train a model. If a port is occupied, check the
existing service rather than starting duplicates or killing unrelated processes.
These commands are for local presentation; do not expose the app publicly.

## Target and protocol

The target is **strictly greater than 70% person-detection mAP50 on the full
548-image VisDrone validation split**. It is not identity-matching accuracy,
count accuracy, or a claim of independent test performance.

The first follow-up experiment reuses the trained 960px checkpoint and evaluates
960, 1280, 1536, 1920, and 2240px inputs. It changes resolution only, keeps batch=1,
and retains the existing full-validation evaluator and counting protocol.
Higher resolutions can increase computation and are not automatically suitable
for real-time deployment. No tracking, identity matching, or deployment runs.

```powershell
cd S:\three_eyed_raven
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m crowd_tracker.detection_sweep_cli --checkpoint runs\experiments\extended_detection_20261012\visdrone_person_960\weights\best.pt
```

Every invocation uses a new experiment directory. Results and progress live under
`runs/experiments/detection_resolution_<timestamp>/`. The experiment does not
replace the dashboard's previous training result or alter existing checkpoints.
`comparison.json` records whether the target was actually reached; `REPORT.md`
includes the full comparison. Failed resolutions are reported, not omitted.

Resolution selection is development-set tuning. A later independent test on
unseen scenes is necessary before claiming generalisation. If resolution alone
plateaus below the target, investigate missed small/occluded people and evaluate
tiling or a larger detector; do not silently change the dataset or metric to
manufacture a passing score.

An optional `--tta` flag enables Ultralytics test-time augmentation for both
detection validation and counting predictions. Keep it in a separately named
experiment, since it changes inference work and latency. This is not additional
training. The report records the setting explicitly.

## Measured follow-up results (12 October 2026)

All runs below use the same trained checkpoint and all 548 validation images.
Count MAE uses the 423-image report partition, with a confidence threshold chosen
on the separate 125-image calibration partition. Lower MAE is better.

| Inference setting | mAP50 | Count MAE (people/image) |
| --- | ---: | ---: |
| 960px | 59.00% | 8.270 |
| 1280px | 62.11% | 7.969 |
| 1536px | 62.22% | 7.823 |
| 1920px | 59.97% | 8.106 |
| 2240px | 56.62% | 7.943 |
| 1536px with test-time augmentation | 62.61% | 7.709 |
| 1920px with test-time augmentation | 61.98% | 7.494 |

The 70% target has **not** been reached. Resolution alone peaks and then worsens.
At 1536px, augmentation reduced measured count-evaluation throughput:
16.72 images/second versus 29.42 without augmentation on this run. These timings
include sequential image loading and progress writes; they are not end-to-end
live-camera FPS or controlled warmed-up GPU benchmarks.

Local detailed reports are under `runs/experiments/detection_resolution_20261012/`
and `runs/experiments/detection_tta_20261012/`. No checkpoint or deployment
configuration was changed. The next controlled experiment should examine missed
small/occluded people, then compare a larger detector or tiled inference on the
same full split. Independent testing remains necessary.
