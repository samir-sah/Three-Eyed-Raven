# Panel demonstration script

## Before the session

1. Start the API: `python -m uvicorn crowd_tracker.api:app --host 127.0.0.1 --port 8000`.
2. Start the dashboard: `cd dashboard; npm run dev`.
3. Open `http://127.0.0.1:3000` and select a completed recorded run.
4. Keep the prepared demo videos and the latest annotated output available locally. Do not depend on an external camera or internet connection during the presentation.

## Five-minute walkthrough

1. **Problem and scope (30 sec).** Explain that the system detects people, assigns *local* per-camera track IDs, counts zone occupancy, and surfaces threshold alerts for operator review.
2. **Video evidence (60 sec).** Play the annotated output. Point out person boxes, local IDs, crowd count, and the configured zone overlay.
3. **Dashboard evidence (60 sec).** Show processed frames, peak count, processing FPS, local-ID count, the playback panel, and the track-health list.
4. **Alert workflow (45 sec).** Open alerts and explain that an alert is a zone-threshold event rather than a claim about an individual.
5. **Multi-camera safety (45 sec).** Show cross-camera candidates. State clearly that they are ranked visual similarities for human review and the system never merges identities automatically.
6. **Evaluation (45 sec).** Open the MOT17 benchmark card and explain MOTA, MOTP, precision/recall, and ID switches. Identify it as a baseline rather than a final benchmark.
7. **Research progress (30 sec).** Show the VisDrone data-preparation/training result and the documented full experiment plan.

## Questions to answer consistently

- **Is this facial recognition?** No. It performs person detection, local tracking, and review-only appearance ranking; it should not be used to identify people.
- **Can it process live cameras?** The capture and source-preflight code supports authorised RTSP/HTTP streams, but the assessed demo uses recorded video for repeatability. Field validation remains required.
- **What does a track ID mean?** It is a temporary local identifier within one camera stream, not a person’s real-world identity.
- **What is still required?** Full multi-epoch detector training, licensed Re-ID data experiments, full benchmark aggregation, and authorised field validation.

## Fallback plan

If the API or dashboard is unavailable, play `annotated.mp4` and show `summary.json`, `alerts.jsonl`, and the saved evaluation JSON. This preserves the key evidence without depending on a network or live camera.
