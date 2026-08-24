# Real-Time Crowd Tracking Using Drones and Surveillance Cameras

## 1. Project definition and delivery boundary

Build an academic, software-first surveillance prototype that ingests one drone-like video feed and one or more CCTV-like video feeds, detects people, maintains **per-camera track IDs**, attempts **cross-camera/aerial-ground person association**, and presents crowd analytics and alerts in a dashboard.

The first demonstrable version should use recorded video and public datasets. Live CCTV/RTSP streams and a real drone are integration work after the analytics pipeline is reliable. The project must not claim biometric identification: it performs appearance-based **person re-identification (Re-ID)** with uncertain matches, and it must not use facial recognition.

### MVP acceptance criteria

- Two recorded streams can be processed concurrently or replayed in a synchronized demo.
- People are detected and assigned stable IDs within each stream.
- The UI shows annotated video, current count, FPS, track trajectories, and zone-density alerts.
- A cross-view matching panel shows candidate matches with a similarity score and a clearly visible confidence/"uncertain" state.
- Evaluation is reproducible on public data and reports detection, tracking, Re-ID, latency, and hardware configuration.

### Non-goals for the initial project

- Autonomous drone navigation or flight control.
- Surveillance-grade identity verification, law-enforcement use, or automatic action against people.
- City-scale distributed deployment, face recognition, or fully reliable matching in dense crowds.

## 2. Recommended architecture

```text
Video sources (files / RTSP) ──> stream worker per camera
                                  ├─ frame validation + timestamps
                                  ├─ YOLO person detector
                                  ├─ ByteTrack tracker (local track ID)
                                  ├─ optional Re-ID embedding extractor
                                  └─ event/metric publisher
                                                │
                        shared track + gallery service
                                  ├─ camera metadata and zones
                                  ├─ time/gate constraints
                                  └─ cross-view candidate ranking
                                                │
                  SQLite/PostgreSQL + media/metrics storage
                                                │
                  FastAPI backend ──> Streamlit or React dashboard
```

**Baseline choices:** Python 3.10/3.11, PyTorch + CUDA, Ultralytics YOLOv8 (person class only), ByteTrack, OpenCV, FastAPI, SQLite for the prototype, Streamlit for the fastest dashboard. Use Docker only after the local pipeline works. DeepSORT is useful as a comparison baseline, but do not build both trackers into the initial critical path.

## 3. Phase plan

| Phase | What it accomplishes | Entry prerequisites | Deliverables / exit criteria |
|---|---|---|---|
| 0. Scope, safety, and success metrics | Converts the report into testable requirements and a feasible academic demo. | Agreement on demo setting, available GPU, team roles. | Requirements, threat/privacy note, metrics sheet, 2-stream demo scenario, risk register. |
| 1. Foundations and environment | Creates a reproducible development base. | Python basics, Git workflow, NVIDIA driver/GPU access or a cloud GPU decision. | Git repo, virtual environment/lock file, CUDA/PyTorch validation, sample-video loader, CI/lint/test baseline. |
| 2. Data and evaluation harness | Establishes data quality and objective testing before model changes. | Dataset licensing reviewed; storage; annotation knowledge. | Dataset inventory, train/validation/test splits, conversion scripts, video metadata schema, evaluator that saves metrics. |
| 3. Person-detection baseline | Detects people reliably in each feed. | Phase 1-2 complete; core deep-learning and image-processing concepts. | YOLO baseline, annotated output video, precision/recall/mAP report, inference-speed benchmark. |
| 4. Single-camera multi-object tracking | Gives stable local identity and trajectories. | Detection output with timestamps; understanding of IoU, Kalman filtering, association. | ByteTrack pipeline, ID-switch/MOTA/HOTA-style report, track lifecycle policy, occlusion test videos. |
| 5. Crowd analytics and alerts | Turns tracks into an operational crowd-monitoring demo. | Calibrated/defined screen zones; agreed alert thresholds. | Counts, occupancy/density by zone, dwell time/flow, threshold alerts, event log. |
| 6. Cross-view Re-ID prototype | Ranks likely same-person candidates across CCTV and aerial feeds, with guardrails. | Strong per-camera tracking; Re-ID dataset and evaluation protocol; embeddings knowledge. | Pretrained Re-ID baseline, gallery/query service, Rank-1 and mAP report, conservative match policy. |
| 7. Multi-stream and live-source integration | Replaces files with synchronized, resilient stream ingestion. | Stable offline pipeline; access to authorized RTSP camera/drone feed. | Stream manager, reconnect/buffer behavior, timestamp alignment, performance/load test. |
| 8. Dashboard, APIs, and persistence | Provides an operator-facing, explainable interface. | Stable event schema and alert logic. | Dashboard, FastAPI endpoints, track/event storage, playback, exported report, role/access stub. |
| 9. Optimization, validation, and release | Produces a defensible final project demo/report. | All modules integrated; test scenes and target device. | End-to-end benchmark, stress test, failure analysis, demo script, installation guide, final report results. |

## 4. Knowledge and skill prerequisites by phase

### Phase 0 — scope and responsible design (3–5 days)

Learn: functional/non-functional requirements, experiment design, basic privacy principles, and limitations of appearance matching.

Decide: a single controlled demonstration area, 2 video sources, maximum target density (~50 people/frame), an alert such as `zone count > threshold`, and performance targets tied to the actual hardware. Obtain permission for every real camera/drone recording; prefer public datasets or consented footage. Retain raw footage only as long as necessary and blur/avoid publishing identifiable faces in demos.

### Phase 1 — engineering foundation (3–5 days)

Learn: Python packaging, virtual environments, Git branches/pull requests, GPU/CUDA compatibility, logging, configuration files, and unit testing.

Technical prerequisites: NVIDIA RTX 3060-class GPU or a cloud equivalent; 16 GB RAM; 512 GB SSD; Python; Git; VS Code. Verify `torch.cuda.is_available()` and run a known model inference before any application code. Keep model weights, datasets, secrets, and recorded raw video out of Git.

### Phase 2 — data and evaluation (1–2 weeks)

Learn: dataset cards/licences, COCO/MOT/Re-ID annotation formats, train/validation/test leakage, and metrics: precision, recall, mAP, FPS, MOTA/IDF1, ID switches, Rank-1, and Re-ID mAP.

Start with VisDrone2019 for aerial detection, MOT17 for tracking, and an aerial-ground Re-ID dataset named in the report (verify its current access terms before use). Do not mix identities/scenes across splits. Record resolution, FPS, camera type, timestamp basis, and consent/licence for every video.

### Phase 3 — detection baseline (1–2 weeks)

Learn: bounding boxes, confidence thresholds, non-maximum suppression, transfer learning, augmentation, and overfitting.

Implementation: use a pretrained YOLOv8 model, filter to class `person`, and benchmark before fine-tuning. Fine-tune only if error analysis shows a material aerial-domain gap. Save the model version, threshold, image size, hardware, and measured FPS with every result.

### Phase 4 — tracking (1–2 weeks)

Learn: detection-to-track association, IoU, motion models/Kalman filters, Hungarian matching, track states, and ID-switch causes.

Implementation: integrate ByteTrack first. Define a tracker interface so DeepSORT can later be run as an experimental comparison. Test missed detections, entry/exit edges, brief occlusion, and similarly dressed people. Never treat a local tracker ID as a real-world identity.

### Phase 5 — analytics and alerts (1 week)

Learn: ROI polygons, counts vs density estimates, temporal smoothing, threshold selection, and false-alert review.

Implementation: start with counts inside manually configured polygon zones, then calculate a rolling average and hold time (for example, alert only when the threshold is exceeded for 10 seconds). Store every alert with source, timestamp, threshold, observed value, and a video/frame reference. Describe it as crowd occupancy unless perspective calibration justifies physical density units.

### Phase 6 — cross-view Re-ID (2–4 weeks; highest research risk)

Learn: embeddings, cosine distance, query/gallery protocol, domain shift, metric learning, Siamese/triplet loss, and calibration.

Implementation sequence:

1. Use a pretrained Re-ID model to extract an embedding for high-quality person crops.
2. Build a query/gallery evaluator and report Rank-1 and mAP on the selected dataset.
3. Apply practical gating before similarity matching: valid camera-pair, feasible travel time, person-box quality, and track duration.
4. Return a ranked candidate list—not a forced match. Auto-link only high-confidence matches; label the remainder `unknown` or require review.
5. Fine-tune on aerial-ground data only after this baseline is measured. A custom Siamese network is an optional comparison experiment, not a prerequisite for an MVP.

This sequencing matters: aerial-to-ground Re-ID is the most difficult part of the proposal and will often perform poorly when people are small, occluded, or differently viewed. The project is still successful if it reports this limitation honestly and demonstrates confidence-aware candidate matching.

### Phase 7 — streaming integration (1–2 weeks)

Learn: RTSP/video codecs, buffering, threading/async queues, clock drift, dropped frames, reconnection, and back-pressure.

Implementation: use recorded files first, then RTSP. Place each stream in its own worker and communicate through bounded queues. Use capture timestamps rather than loop counters; report both processing FPS and end-to-end latency. A drone feed may be represented by a recorded aerial video if hardware access or flight permission is unavailable.

### Phase 8 — dashboard and persistence (1–2 weeks)

Learn: REST APIs, database schemas, UI state, data retention, and basic authentication/authorization concepts.

Implementation: FastAPI owns the backend contract; Streamlit renders the initial UI. Store camera, track, match-candidate, metric, and alert records. The dashboard must distinguish local IDs from cross-view candidates and display uncertainty. Do not expose video streams without access controls in a real deployment.

### Phase 9 — validation and hand-off (1–2 weeks)

Learn: profiling, reproducible experiments, load testing, error analysis, and technical documentation.

Validate distinct scenarios: sparse scene, moderate crowd, low light, partial occlusion, camera hand-off, aerial small-person case, stream disconnect, and alert threshold boundary. Measure against the targets in the report, explain deviations, and record known failure cases.

## 5. Work breakdown / implementation backlog

### Milestone A: offline single-camera demonstrator

1. Create repository structure: `src/ingest`, `src/detect`, `src/track`, `src/analytics`, `src/reid`, `src/api`, `tests`, `configs`, `scripts`.
2. Add YAML configuration for sources, model paths, thresholds, zones, output paths, and runtime device.
3. Implement a video reader that emits `{camera_id, frame_id, timestamp, frame}`.
4. Add YOLO inference and a visual overlay writer.
5. Integrate ByteTrack and emit a normalized `TrackObservation` event.
6. Add tests for configuration validation, frame timestamps, zone inclusion, and track-event serialization.

**Demo:** one video with person boxes, local IDs, trajectories, count, and CSV/SQLite metrics.

### Milestone B: analytics and two-source playback

1. Define camera metadata and manually draw zones per camera.
2. Implement zone counts, rolling occupancy, entry/exit flow, and alert state machine.
3. Run two separate workers on a CCTV-like and an aerial-like recorded feed.
4. Add a simple stream health view: input FPS, inference FPS, queue depth, last-frame time.

**Demo:** synchronized two-panel playback with alerts and event history.

### Milestone C: Re-ID experiment and controlled hand-off

1. Add crop-quality checks and embedding extraction from selected track crops.
2. Save a compact gallery keyed by camera, local track, time interval, and quality score.
3. Implement cosine ranking plus temporal/camera-pair gates.
4. Build an offline evaluator and compare generic pretrained vs fine-tuned model (if time permits).
5. Present top-k candidates and only persist high-confidence links.

**Demo:** selected query track shows candidate tracks in the other stream, score, thumbnail, and `matched/uncertain/rejected` state.

### Milestone D: live integration and finalization

1. Replace file reader with RTSP adapters and reconnection tests.
2. Add FastAPI service, SQLite persistence, and Streamlit dashboard.
3. Profile GPU/CPU and reduce latency: batching only where it does not harm real-time response, half precision where supported, resize policy, frame sampling under overload.
4. Package run instructions, system diagram, dataset register, test results, and a scripted 5–7 minute demo.

## 6. Team roles (four-person project)

| Role | Primary ownership | Secondary/review responsibility |
|---|---|---|
| CV detection/tracking engineer | YOLO, ByteTrack, performance profiling | data/evaluation |
| Re-ID and research engineer | embeddings, gallery/query evaluation, experiments | tracker integration |
| Backend/streaming engineer | video ingestion, RTSP, event schema, database, APIs | deployment |
| Dashboard/QA engineer | UI, alerts, test scenarios, documentation, demo | metrics/reporting |

All members should use code review, issue tracking, a shared experiment log, and weekly integrated demos. Assign a single owner to each module, but make each critical interface testable by another member.

## 7. Dependencies, risks, and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Aerial-ground Re-ID accuracy is low | Cross-view claim is weak | Deliver ranked, confidence-gated candidates; present measured limits; prioritize per-camera tracking. |
| GPU or CUDA mismatch | Blocks development | Verify environment in Phase 1; keep a small CPU demo mode; use approved cloud GPU if needed. |
| Dataset access/licensing issues | Delays experiments | Register datasets early; retain dataset cards; use alternative public data or consented recordings. |
| Real drone/RTSP feed unavailable | Blocks live demo | Treat recorded aerial/CCTV videos as first-class demo sources; schedule hardware integration last. |
| Dense scenes cause ID switches | Analytics becomes unreliable | Test moderate density first; tune detector/tracker; surface track confidence and do not overstate performance. |
| Privacy/security concern | Project risk | No face recognition; authorization and retention rules; blur media in presentations; never use the system for decisions about individuals. |
| Dashboard work starts too early | Delays core system | Build only a lightweight visualizer until detection/tracking metrics are stable. |

## 8. Suggested 16-week schedule

| Weeks | Primary outcome |
|---|---|
| 1–2 | Phase 0–1: scope, environment, repository, sample streams |
| 3–4 | Phase 2: dataset register, evaluator, initial baselines |
| 5–6 | Phase 3: detection benchmark and tuned inference pipeline |
| 7–8 | Phase 4: single-camera tracking and tracker evaluation |
| 9 | Phase 5: zones, counts, alerts |
| 10–12 | Phase 6: cross-view Re-ID baseline, gates, evaluation |
| 13 | Phase 7: two-stream/live-source integration |
| 14 | Phase 8: dashboard, API, persistence |
| 15–16 | Phase 9: profiling, stress tests, report, final demo |

If the calendar is shorter, retain Phases 0–5 as the core project and make Phase 6 an evaluated prototype rather than a promised end-to-end identity continuity feature.

## 9. Definition of done

The final submission is complete when a clean machine/environment can follow the README to run the recorded two-stream demonstration; the code produces saved annotated outputs and a dashboard; every major claim has a metric and test dataset/video; the report separates measured results from targets; and the limitations of Re-ID, crowd density, lighting, occlusion, and privacy are explicit.
