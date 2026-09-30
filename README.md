# Crowd Tracking Baseline

This is the first runnable implementation slice of the project: one video source is processed with YOLO person detection and ByteTrack local tracking. It produces an annotated MP4 (optional), per-frame JSONL events, and a run summary.

## Current scope

- Recorded video input only; RTSP/CCTV and drone feeds come after this offline baseline is evaluated.
- `person` detections only (COCO class 0).
- Stable **local** track IDs within one camera stream. A track ID is not a verified real-world identity.
- Polygon-zone occupancy and threshold alerts.

Cross-camera person re-identification and the operator dashboard are deliberately not included in this first slice. They depend on reliable per-camera tracking and a measured evaluation dataset.

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

## Next implementation steps

1. Evaluate detection and tracking on VisDrone/MOT17 rather than relying on a visual demo.
2. Add a Re-ID experiment service that returns confidence-ranked candidates, never forced identities.
3. Add FastAPI persistence and a dashboard after the event schema is stable.

## Two-stream baseline

`multi_config.example.json` runs an aerial and a ground-level source in an interleaved loop. Each camera has its own ByteTrack instance and therefore its own local ID namespace; this is deliberate, because cross-camera identity linking belongs to the later Re-ID phase.

```bat
copy multi_config.example.json multi_config.json
set PYTHONPATH=src
python -m crowd_tracker.multi_cli --config multi_config.json
```

Each camera gets its own annotated video, observations, alerts, and summary under `artifacts/two_stream_demo`.

The same run emits `reid_candidates.json`: an explainable HSV appearance-matching baseline that ranks possible cross-camera matches. Its output is explicitly review-only; no local IDs are automatically merged and it is not biometric identity verification. A trained aerial-ground Re-ID model is the next research upgrade after this baseline is evaluated.

## Dashboard preview frames

The annotated MP4 is created with OpenCV's FMP4 codec, which some browsers cannot play. Generate a browser-safe frame sequence for the Next.js dashboard with:

```bat
python scripts\export_preview_frames.py --run live_preview
```
