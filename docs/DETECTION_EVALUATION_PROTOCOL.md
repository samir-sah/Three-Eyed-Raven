# Detection and crowd-count evaluation

## Purpose and definitions

Evaluate anonymous person boxes and per-image counts separately from the existing tracker and cross-camera features. A camera-local trajectory ID is not a unique-person count. Counts from overlapping cameras are not added as unique occupancy.

For the MOT17 development audit, a target is an annotation with positive mark and class 1 (pedestrian). Include all supplied visibility levels, including occluded targets. Use the dataset's supplied full-body box; do not invent a new visible-body rule. Non-target and unmarked annotations are ignored regions. These counts therefore concern the benchmark's annotated target category, not every visible human category.

Predictions match targets in descending confidence, one prediction per target, at IoU >= .5. An unmatched prediction overlapping an ignored region by at least .5 of its own area is neutral. Target matches take precedence. AP repeats this matching at IoU .50 through .95, using Ultralytics' 101-point integration. This is a documented custom detection audit, not official MOTChallenge/TrackEval scoring.

## Fixed ground audit

Convert MOT's 1-based box origins to the detector's 0-based image coordinates before matching. Preserve the initial audit run as preliminary; only the corrected run is shown on the dashboard.

- Source: the seven MOT17 public training scenes, using only `*-FRCNN` folders to avoid repeating the same scene under three detector labels. Their images are input; supplied FRCNN detections are not used.
- Sample frame 1 and one frame per source second throughout each scene. This is a sampled audit, not evaluation of every video frame.
- Calibration scenes: 02, 04, 05, 10. Report scenes: 09, 11, 13. Scene boundaries stay intact.
- Freeze the image manifest and SHA-256 hashes of images, annotations and each model checkpoint.
- Compare COCO YOLO11n (upload detector), historical VisDrone 50-epoch checkpoint, extended VisDrone YOLO11n and exploratory VisDrone YOLO11s.
- Matched default resolution: 640px, class 0, no augmentation, NMS IoU .5, maximum 300 detections. Minimum .001 confidence for AP; .35 operating confidence to match the upload setting.
- Report precision/recall/F1 and count MAE, RMSE and signed bias at .35. Positive bias means overcounting; negative means undercounting. Retain raw counts and ignored predictions.
- Separately select a confidence threshold using maximum calibration F1 over .05 to .65, stepping .05. Evaluate that threshold unchanged on report scenes. The threshold optimizing F1 need not optimize counts.
- Record per-scene results, small/medium/large recall (original box areas <32², <96², >=96²), and largest miss counts for inspection.
- Speed covers image decoding and detector prediction after warm-up. It excludes metric calculation and website/video rendering, so it is not end-to-end video FPS.

## Interpretation

MOT17 report scenes are separate from threshold tuning, but this dataset was already inspected in previous project experiments. Results are development evidence, not a previously untouched final test. They do not establish performance on uploaded videos or prove aerial-to-ground generalization. No checkpoint or threshold is deployed automatically.

The previous VisDrone experiments also used validation data for selection. Keep the better-performing nano checkpoint as the preferred tested aerial candidate and preserve all unsuccessful checkpoints/logs.

## Independent final test still required

Collect permitted representative clips from several aerial and ground scenes. Describe crowd density, person size, lighting, blur and camera angle. Label under a written per-view guide, with a second reviewer checking a subset. Public datasets must retain their original annotation conventions and ignore handling; never combine incompatible point/box scores under one metric name.

Keep whole scenes/videos in train, validation or final test. Do not split neighbouring frames at random. Freeze the final test before model selection; inspect it once after settings are frozen. Head-point annotations may support a separate dense-crowd counting task, not box AP. Obtain enough independent scenes to estimate uncertainty; many neighbouring frames do not substitute for scene diversity.

## Reproduce on the laptop

```powershell
cd S:\three_eyed_raven
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m crowd_tracker.ground_detection_cli --name ground_detection_new_run
```

Use a new name for each experiment; existing runs are never overwritten. Add `--smoke` to exercise the pipeline with one image from each scene. Smoke results are not accuracy evidence. Results are saved beneath `runs/experiments/<name>` with `manifest.json`, `comparison.json`, `REPORT.md`, per-model reports and anonymous box predictions. Only non-smoke runs update `ground_active.json` for the dashboard.
