# Three Eyed Raven — Crowd Tracking Platform

This runnable project processes recorded aerial and CCTV video with YOLO person detection and ByteTrack local tracking. It produces annotated outputs, per-frame JSONL events, zone-threshold alerts, cross-camera review candidates, and a local dashboard for demonstration.

## Current scope

- Recorded video input only; live RTSP/CCTV and drone feeds come after this offline baseline is evaluated.
- `person` detections only (COCO class 0).
- Stable **local** track IDs within one camera stream. A track ID is not a verified real-world identity.
- Polygon-zone occupancy and threshold alerts.
- Cross-camera appearance ranking is review-only; it never merges IDs or asserts a real-world identity.
- A Next.js dashboard reads the local artifacts and presents playback, metrics, alerts, and review evidence.

## Setup

Use Python 3.10 or later. An NVIDIA CUDA-enabled PyTorch installation is recommended for real-time speed.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy `config.example.json` to a local configuration (for example, `config.json`), set `source` to a video file, then run:

```powershell
$env:PYTHONPATH = "src"
python -m crowd_tracker --config config.json
```

On first run, Ultralytics downloads the selected model weight (`yolo11n.pt` by default). Use `"device": "cpu"` when no CUDA GPU is available; use `"auto"` to let Ultralytics select it.

## Outputs

The configured `output_dir` contains:

- `annotated.mp4` — video with person boxes, local IDs, count, and zone alerts.
- `observations.jsonl` — one event per tracked person per processed frame.
- `alerts.jsonl` — threshold-crossing alerts, deduplicated while a zone stays above its threshold.
- `summary.json` — source metadata, throughput, total unique local IDs, and configuration snapshot.

## Verification

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

## Tracking benchmark evaluation

Use a labeled JSONL file with stable ground-truth IDs to evaluate one camera sequence. Prediction records use the project's existing `observations.jsonl` format. The evaluator reports precision, recall, F1, MOTA, MOTP, false positives/negatives, and local-ID switches at the chosen IoU threshold.

```bat
set PYTHONPATH=src
python -m crowd_tracker.tracking_eval_cli --ground-truth examples\tracking_ground_truth.jsonl --predictions examples\tracking_predictions.jsonl --output artifacts\benchmark_example.json
```

The bundled files are a format example only. Report real results only after evaluating against a labeled benchmark such as MOT17 or VisDrone-MOT; do not treat demo footage as ground truth.

## Next implementation steps

1. Evaluate detection and tracking on VisDrone/MOT17 rather than relying on a visual demo.
2. Add a Re-ID experiment service that returns confidence-ranked candidates, never forced identities.
3. Add a persistent API service and authenticated operator workflow after the event schema is stable.

## Two-stream baseline

`multi_config.example.json` runs an aerial and a ground-level source in an interleaved loop. Each camera has its own ByteTrack instance and therefore its own local ID namespace; this is deliberate, because cross-camera identity linking belongs to the later Re-ID phase.

```bat
copy multi_config.example.json multi_config.json
set PYTHONPATH=src
python -m crowd_tracker.multi_cli --config multi_config.json
```

Each camera gets its own annotated video, observations, alerts, and summary under `artifacts/two_stream_demo`.

The sample two-stream configuration includes a full-frame observation zone for each camera and a demo threshold of three local tracks. In a deployment, replace these rectangles and thresholds with site-specific zones after calibrating each camera view.

The same run emits `reid_candidates.json`: an explainable HSV appearance-matching baseline that ranks possible cross-camera matches. Its output is explicitly review-only; no local IDs are automatically merged and it is not biometric identity verification. A trained aerial-ground Re-ID model is the next research upgrade after this baseline is evaluated.

For a labeled aerial-ground dataset, evaluate exported query/gallery embeddings with standard Rank-1, Rank-5, Rank-10, and mAP metrics:

```bat
set PYTHONPATH=src
python -m crowd_tracker.reid_eval_cli --manifest reid_eval.example.json
```

The included manifest is only a schema example; real metrics require a dataset with known same-person labels across camera views.

## Dashboard preview frames

The annotated MP4 is created with OpenCV's FMP4 codec, which some browsers cannot play. Generate a browser-safe frame sequence for the Next.js dashboard with:

```bat
python scripts\export_preview_frames.py --run live_preview
```
