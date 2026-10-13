# Laptop improvement session — 13 October 2026

## What is deployed for new uploads

- Anonymous per-frame detection/counting only. The legacy ByteTrack/Re-ID pipeline and its detector configurations are unchanged.
- Aerial: `runs/experiments/extended_detection_20261012/visdrone_person_960/weights/best.pt`, YOLO11n, 960px. The later VisDrone-trained YOLO11s was worse and is not used.
- Ground: COCO-pretrained `yolo11s.pt`, 960px, confidence .35. Ground-level/elevated footage uploaded in the aerial slot can explicitly select this general detector too.
- Auto CUDA/CPU selection; one camera's model is loaded at a time. Upload jobs are serialized and rejected while the active training/evaluation process is alive.
- H.264/yuv420p/fast-start browser videos, all processed frames, maximum output long edge 1280px, original aspect ratio. White anonymous boxes and a compact count label, no IDs/crops/embeddings.
- Shared nominal clip time, seeking, replay and per-view fullscreen. This does not assert clock synchronization between cameras. Shorter clips hold their last image.
- Every frame is logged in `counts.jsonl`, including zero counts. Peak/trough-preserving charts use actual relative timestamps. Counts are not added across overlapping views, and are not unique visitors.
- No alert limit configured; the interface explicitly says alerts are off.

## Matched ground-camera profile comparison

Saved evidence: `runs/experiments/laptop_ground_profiles_20261013_complete`.
215 images sampled approximately once per second from seven MOT17 public training scenes. 137 images from four calibration scenes; 78 report images from three other scenes, containing 965 annotated pedestrian boxes. The same custom anonymous detection/ignore protocol and images are used for all profiles. No tracker or identity column is evaluated.

| Profile | Report mAP50 | Recall at .35 | Precision at .35 | Count MAE |
|---|---|---|---|---|
| COCO YOLO11n · 640px | 57.40% | 43.21% | 91.05% | 6.55 |
| COCO YOLO11s · 640px | 61.86% | 48.91% | 88.56% | 5.62 |
| COCO YOLO11n · 960px | 62.30% | 50.47% | 89.36% | 5.38 |
| COCO YOLO11s · 960px | 64.53% | 53.68% | 86.33% | 4.73 |

The selected profile improves mAP50 by 7.13 percentage points and recall by 10.47 points, with about 27.8% lower count MAE, versus the previous upload default. Precision is 4.72 points lower: more people are found but more incorrect boxes are admitted. It is not proof of 64.53% accuracy on arbitrary videos, and it does not establish 70% on independent test data. The report scenes were already inspected during development and profile selection; a future untouched test set is required.

## Video integration check

`artifacts/upload_20261013_anonymous` reprocesses the existing uploaded pair without altering the originals or previous results. It logs 74 aerial-slot frames and 600 ground-slot frames, and exports 2.47s/12s browser videos. The ground-slot detector peak is 29. The aerial-slot detector peak is only 2: that clip is a ground-level view, illustrating why the viewpoint selector is necessary. These are detector outputs, not manually verified ground truth. Do not claim that a higher displayed count alone proves better accuracy.

Stored footage/output selection is preserved across modes and Refresh. Choose the newer analysis explicitly through Change footage; the app does not silently replace a pinned earlier result. Earlier outputs of that same pair remain selectable under Technical details → Saved analyses. Check the aerial profile before uploading new clips.

## Ground-domain training pilot

`runs/experiments/ground_pilot_20261013` starts from COCO YOLO11s, at 960px, batch 1, workers 0, with a 0.6-hour training budget. It uses 230 anonymous images from scenes 02/04/05 and 44 validation images from scene 10, sampled about twice per second. Report scenes 09/11/13 are not used for training or checkpoint selection, but they were already inspected during previous development. Excluded categories become background during YOLO training because its loss does not support MOT ignore regions; occluded full-body targets remain. This is a small scene-separated pilot, not full-dataset ground training.

The worker saves best/latest checkpoints and compares its candidate against the deployed COCO YOLO11s at matched 960px/.35 settings. No pilot checkpoint is automatically promoted. The training validation mAP and final custom report mAP are different measurements. Completion results appear in its `comparison.json`/`REPORT.md` and the Training page.

The first profile process was interrupted when its short-lived sandbox ended; its partial predictions remain in `laptop_ground_profiles_20261013`. The completed comparison was rerun separately outside that sandbox. Neither partial files nor checkpoints were deleted.

## Checks and operational notes

Python unit tests cover anonymous box clipping, configuration restrictions, scene separation, normalized labels, real browser video encoding, preservation of failures and refusal to overwrite existing outputs. JavaScript tests cover pinned footage, correct count summaries, chart peak retention and byte ranges. The production build succeeds; browser checks verify actual video decoding, playback/end behavior and persistence when changing sections.

The app and training computer must remain powered. Stop training before launching other GPU-heavy tasks. An interrupted upload leaves a lock file in `data/uploads/processing.lock`; inspect its recorded process and job before clearing it. The system never automatically removes a potentially active job's lock.
