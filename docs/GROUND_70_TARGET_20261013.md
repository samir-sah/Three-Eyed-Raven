# Ground detection: the 70% development target

Date: 13 October 2026. Target: strictly greater than 70% **anonymous ground detection mAP50**. This is not a tracking, cross-camera identity, aerial-detection or all-video accuracy claim.

## Result and decision

The frozen calibration-selected `yolo11x_1536_tta` profile reached **70.9642% mAP50** on the unchanged development report. The previous YOLO11s/960 baseline reproduced **64.5255%**. A separate CPU audit recalculated AP and counts from saved predictions and matched both results exactly.

| Same 78 report images, 965 marked pedestrian boxes | Previous YOLO11s/960 | Approved YOLO11x/1536 multi-scale |
|---|---:|---:|
| mAP50 | 64.53% | **70.96%** |
| mAP50–95 | 36.94% | 41.36% |
| Precision at confidence .35 | 86.33% | 81.33% |
| Recall at confidence .35 | 53.68% | 65.91% |
| F1 at confidence .35 | 66.20% | 72.81% |
| Adjusted count MAE, people/image | 4.73 | 2.83 |
| Raw count MAE, people/image | 4.47 | 2.90 |
| Decode + prediction FPS, RTX 3050 laptop | 35.09 | 3.47 |

The improvement is **6.44 percentage points of mAP50**, 12.23 points of recall, and about 40% lower adjusted count MAE. Precision declined by 5 points: the model recovers more people but also makes more false detections. Aggregate precision is not a per-scene guarantee (scene 11: 71.52%).

Approve this profile for **new anonymous ground uploads only**, because the user explicitly prioritizes accuracy over speed. This is an offline accuracy-first setting, roughly ten times slower than the baseline. Keep the previous weights and all logs; do not promote these models into legacy tracking, embeddings or cross-camera matching. True aerial uploads retain the stronger previous VisDrone YOLO11n checkpoint. Existing videos are not silently reprocessed or switched.

## What changed

No new model was trained during this search. Official COCO-pretrained YOLO11 and YOLO26 weights were tested with larger input resolution, FP16 and optional multi-scale test-time augmentation. A larger detector alone was insufficient. The first fixed search selected YOLO11s/1536 multi-scale and reached 69.17%; the second selected YOLO11x/1536 and crossed 70%.

Approved profile:

```json
{
  "name": "yolo11x_1536_tta",
  "checkpoint": "yolo11x.pt",
  "imgsz": 1536,
  "tile_size": 0,
  "augment": true,
  "fp16": true,
  "end2end": false
}
```

Inference uses class 0 (person), NMS IoU .5, up to 300 boxes per frame, and .001 prediction confidence for AP evaluation. Upload counts filter the exact same detector output at .35. Multi-scale includes resized/flipped passes; it is not extra training. The official checkpoint SHA-256 is `7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24`.

`ground_detection_policy.json` approves the anonymous-only profile. The uploader checks its schema and checkpoint hash before creating an analysis. Invalid/missing weights fail explicitly. The separate terminal example contains the same profile; CPU execution records FP32 instead of claiming measured FP16.

## Measurement discipline

- Original corrected manifest and images are unchanged: 137 calibration images from MOT17 scenes 02/04/05/10; 78 report images from 09/11/13. Image and annotation hashes were verified. Calibration/report image hashes do not overlap.
- Fixed standard search: 15 profiles. Fixed advanced search: 18 profiles. Both plans were saved before inference. Profiles use fresh model instances so precision or YOLO26 fusion state cannot leak between settings.
- Select the highest **calibration mAP50**; exact ties use lower fixed-.35 adjusted count MAE and then speed. Advanced winner calibration AP was 68.7728%.
- Save `selection.json` with profile, weight SHA and confidence threshold **before report predictions**. Only baseline and frozen winner are scored on report images. No report-driven reselection within either suite.
- Same target class, full-body boxes, coordinate conversion, ignored-region handling, IoU matching, confidence ordering and 101-point AP integration for both rows. AP uses IoU .50; mAP50–95 averages .50:.05:.95.
- Fixed-.35 comparisons and calibration-selected-threshold comparisons are retained. The advanced winner's selected threshold happened to be .35; no threshold change was needed to cross the AP target.

### Limits on interpretation

**These public MOT17 training scenes were already repeatedly inspected by this project.** Sequential experiment design and tuning make the report development evidence, not an independent final test, even though each suite freezes its winner before report scoring. Do not describe this as a MOTChallenge leaderboard score, general 70.96% correctness, aerial improvement, unique-person count or identity accuracy.

mAP50 summarizes detection over confidence thresholds; recall at the deployed threshold is 65.91%, not 70.96%. Adjusted count error neutralizes ignored-region detections using ground truth that is unavailable in uploaded video. Raw count error includes those detections but compares to marked pedestrians, not a verified annotation of every visible human. Count mistakes can cancel each other; inspect precision and recall as well.

Speed includes image reading/decoding and detector inference, excludes video encoding, web rendering and playback. Actual upload throughput can be lower. Source video playback remains at its nominal FPS. CPU results are not interchangeable with the measured CUDA FP16 profile.

## Reproduction and preserved evidence

Runtime: Python 3.12.10, PyTorch 2.11.0+cu128, Ultralytics 8.4.127, NVIDIA RTX 3050 Laptop GPU, Windows. Source weights are the official Ultralytics assets v8.4.0 aliases; they are not committed to Git.

From the project root, with the corrected MOT17 manifest/data and all required official weights already present:

```powershell
$env:PYTHONPATH = "src"
python -m crowd_tracker.score_target_cli --name ground_score_target_reproduction --suite advanced
```

Do not reuse an existing experiment name or launch concurrent GPU jobs. `runs/experiments/ground_score_target_20261013_v2/` preserves the plan, manifest, selection, per-profile calibration JSON/predictions, report predictions, comparison JSON, report, status, approval decision and demonstration config. `ground_score_target_20261013/` preserves the earlier 69.17% search; failed/older training experiments remain intact.

To verify generalization, next collect consented ground scenes not used in training or any tuning, freeze annotations and scoring, and evaluate this exact profile once. The aerial side still requires separate improvements and its own untouched evaluation data. The high-end desktop plan remains deferred.

## Uploaded-video verification

Reprocessed the existing clips into a separate `upload_20261013_accuracy` analysis, preserving originals, previous outputs and the user's pinned selection. Visual inspection had established that camera A is **ground-level**, and camera B is **elevated CCTV**, not a true drone pair. Both correctly use this general ground profile.

- Ground-level clip: 74 decoded/logged/exported frames, 2.47 seconds, peak estimate 24 people.
- Elevated CCTV clip: 600 decoded/logged/exported frames, 12 seconds, peak estimate 53 people.
- Combined processing: 674 frames in 230.14 seconds, **2.93 FPS including encoding/initialization**.
- These counts are detector estimates without human ground truth. A higher count does not itself prove greater accuracy. No identities or cross-camera association were produced.

Verification: 80 Python tests and 24 dashboard tests passed; the final production build passed. Both videos decoded in the browser without media errors, played to the end and sought back successfully; the video route returned HTTP 206 byte ranges. Exported frame counts exactly matched sequential count-log rows (74 and 600), per-row bounding boxes, and summary peaks. The original pinned analysis was restored after the playback test.
