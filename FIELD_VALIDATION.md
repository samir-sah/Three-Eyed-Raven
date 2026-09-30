# Field Validation Checklist

Use this checklist before presenting a real-camera result as validated.

## 1. Source and privacy

- Obtain authorization for each drone or CCTV source.
- Store stream credentials only in ignored local configuration or a secret manager.
- Confirm the retention period and who may access recorded output.
- Run the preflight command before starting inference:

```bat
set PYTHONPATH=src
python -m crowd_tracker.preflight_cli --source "rtsp://camera-host:554/stream"
```

## 2. Camera calibration

- Record the frame resolution returned by preflight.
- Draw each polygon zone against that exact resolution.
- Choose the occupancy threshold with the site owner; the included demo threshold is not a field recommendation.
- Test the threshold with staged or authorised observed traffic.

## 3. Performance measurement

- Capture at least one representative, labelled sequence per camera view.
- Convert MOTChallenge-style labels with `crowd_tracker.mot_import_cli` where applicable.
- Run `crowd_tracker.tracking_eval_cli` and preserve the resulting JSON report.
- Report GPU, model, resolution, stride, and measured processing FPS alongside metrics.

## 4. Cross-camera review

- Use matched, consented, labelled multi-view data for any Re-ID evaluation.
- Report Rank-1/mAP only from those labels.
- Keep the current appearance baseline in manual-review mode; never merge IDs or make decisions about people from it.

## 5. Release decision

- Demonstration-ready: all local tests pass, source preflight passes, and dashboard/API are healthy.
- Field-validated: the above plus authorised camera calibration and measured labeled-data results.
- Production-ready: additionally complete a security review, access control, retention policy, monitoring, backup/recovery, and operational ownership review.
