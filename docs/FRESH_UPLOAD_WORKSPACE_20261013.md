# Fresh-upload dashboard and playback fixes

13 October 2026.

## User-facing behaviour

- No bundled sample, legacy tracking output or old anonymous analysis is listed. Historical source videos, checkpoints and artifacts were **not deleted**.
- `dashboard_workspace.json` identifies a fresh workspace. Upload jobs and anonymous config snapshots carry the marker. Only matching, completed anonymous uploads with coherent job/source metadata are exposed by `/api/dashboard`.
- The browser's new selection key is workspace-scoped. A stale old pin cannot reload a demo. Starting another upload clears the displayed result until that analysis completes; completion selects its exact run.
- Overview displays per-camera peak estimates and uploaded filenames. Playback alone displays the two videos. Crowd activity has per-camera timelines; Alerts explicitly says that no threshold is configured; Technical details records the model actually used. Model comparisons are separate development research, not accuracy for uploaded footage.
- Sidebar logo/tagline spacing no longer relies on a negative margin. Branding and navigation do not overlap.
- Both video elements have separate play/pause, seek, explicit replay and fullscreen controls. No autoplay, no looping and no shared camera timer. Each video stops at its own end. Leaving Playback pauses both without resetting positions; returning never auto-resumes.
- The uploader defaults to the full video. A frame limit is optional and clearly identifies partial analysis. Both filenames and actual detector settings are linked to the completed upload.

## Detection scope

The measured policy remains unchanged: ground/elevated footage uses approved YOLO11x/1536 multi-scale CUDA FP16; true aerial footage uses the saved VisDrone-trained YOLO11n/960 checkpoint. This workflow is anonymous image-level detection and counting only. No identity matching, embedding extraction or cross-camera link improvement was added. The 70.96% ground mAP50 remains a development benchmark, not a guarantee for a user's clips.

## Recoverability and QA

Old data remains in `data/uploads` and `artifacts`; the dashboard marker changes visibility, not storage. Do not copy an old summary/job into the new workspace or relabel an old benchmark as a new upload. Invalid or missing markers fail closed instead of rediscovering demos.

Synthetic clips used for upload/playback QA contain labelled text and no people; they are not an accuracy test. Their media/output/logs are preserved separately and excluded from the final workspace. The final dashboard is handed off empty, ready for the user's own new files.

## Verification

- 80 Python regression tests and 43 dashboard tests passed. Production compilation passed.
- Real browser file-chooser upload completed through the actual API/worker, not a fabricated summary. QA run `upload_20261013172928_e063a1` used the approved YOLO11x/1536 profile for both general views, with no frame limit.
- 12 and 16 source frames matched the encoded video frame counts and sequential count-log rows. Blank synthetic frames correctly had zero detections; no old sample count was reused.
- Browser confirmed both videos initially paused, autoplay/loop disabled, correct clip durations/source URLs, isolated camera A and B play/pause, explicit replay after ending, camera B seek without changing A, and pause-with-position-preservation when navigating away.
- Fullscreen has an explicit Exit toggle, verified in the browser. Desktop branding and the 390px mobile menu were checked; selecting a mobile sidebar section closed the drawer and displayed its correct panel. The temporary viewport was reset.
- Overview showed the new filenames/counts; Crowd activity showed separate charts; Alerts showed off; Technical details showed the actual YOLO11x/1536 multi-scale settings. No browser console errors/warnings were captured.
- The QA job's original metadata is saved as `job.qa_original.json`; its active marker was moved to a QA-only UUID, so it is excluded without deleting any file. The production workspace marker was not rotated and any genuine new user upload remains unaffected.
- The first sandboxed Python test attempt hung in a FastAPI test; it was stopped and rerun in the host runtime, where all 80 tests passed in 1.23 seconds. No training or detection task was interrupted.
