# Implementation status

Last verified: 30 September 2026

## Delivered and verified

| Roadmap area | Implemented evidence | Status |
| --- | --- | --- |
| Environment and reproducibility | Python virtual environment, CUDA PyTorch, pinned requirements, tests, Docker compose, README and dataset instructions | Complete locally |
| Video ingestion and preprocessing | File/RTSP sources, frame preprocessing, reconnect settings, bounded threaded capture, timestamp pairing primitives | Code complete; live field validation pending an authorised camera |
| Detection | YOLO person detection; VisDrone DET train/validation preparation; 50-epoch CUDA fine-tuning run with saved best checkpoint | Complete locally; aerial-to-ground generalisation requires further research |
| Local multi-object tracking | ByteTrack per camera, annotated video, JSONL observations, local-ID safeguards | Complete baseline |
| MOT evaluation | MOT ground-truth import, direct image-sequence run, and batch aggregation for MOTA/MOTP/IDF1/HOTA/ID switches | Complete locally on all seven available MOT17 FRCNN train sequences |
| Cross-camera review | Explainable HSV appearance ranking, candidate crops, trained Siamese Re-ID training/evaluation infrastructure | Review-only baseline complete; trained model awaits licensed data |
| Dashboard and persistence | Next.js dashboard, playback, alerts, profiling, benchmark cards; FastAPI and SQLite run index | Complete local demo |
| Integration and operations | Multiple recorded streams, latency profiling, source preflight, API-key option, container definitions | Complete local demo; deployment validation pending |

## Measured baselines

These figures are reproducible baselines, **not final project claims**.

| Experiment | Data and configuration | Result |
| --- | --- | --- |
| Tracking | MOT17-02-FRCNN, first 120 frames, COCO-pretrained YOLO11n + ByteTrack | 16.95 processing FPS; precision 0.842; recall 0.206; F1 0.331; MOTA 0.167; MOTP 0.805; 2 ID switches |
| Detector smoke training | VisDrone DET person labels, YOLO11n, 640 px, batch 4, 1 epoch on RTX 3050 Laptop GPU | mAP50 0.204; mAP50-95 0.062; checkpoint saved to `runs/detect/outputs/detection/visdrone_person_smoke/weights/best.pt` |
| Detector full training | VisDrone DET person labels, YOLO11n, 640 px, batch 4, 50 epochs on RTX 3050 Laptop GPU | precision 0.623; recall 0.433; mAP50 0.476; mAP50-95 0.184; best checkpoint saved to `runs/detect/runs/detect/visdrone_person_50e/weights/best.pt` |
| Trained-model MOT benchmark | All 7 MOT17 FRCNN train sequences, VisDrone-trained YOLO11n + ByteTrack | 112,297 GT boxes; precision 0.909; recall 0.171; F1 0.288; MOTA 0.151; MOTP 0.769; IDF1 0.219; internal HOTA 0.173; 311 ID switches |

The detector smoke run established that CUDA, labels, validation, logging, and checkpoint export work. The completed 50-epoch experiment improves aerial validation mAP substantially, while the full MOT17 result exposes an expected aerial-to-ground domain gap; these findings should be reported rather than hidden.

## Remaining work that cannot be truthfully marked complete yet

1. **Trained cross-camera Re-ID.** PRAI-1581, CARGO, and AG-ReID require their respective access/licence conditions. Obtain them legally, prepare a manifest, train the supplied Siamese model, then report Rank-1/5/10 and mAP. The application must keep matching as operator review, never automatic identity confirmation.
2. **Authorised live field validation.** Calibrate zones, source credentials, retention policy, alert thresholds, permission notices, and test with real drone/CCTV feeds only after approval.
3. **Academic finalisation.** Replace the interim baseline values in the report/slides with the completed detector and MOT17 results, cite data licences, include the aerial-to-ground limitation, and rehearse the live-demo fallback.

### Metric reporting note

The repository's HOTA implementation is an internal, transparent calculation over IoU thresholds from 0.05 to 0.95. It is useful for consistent local comparisons, but before publishing a thesis, paper, or final external benchmark table, compare the same MOT17 predictions with the official TrackEval implementation and report any methodological differences.

## Exact command for the next experiment

```powershell
$env:PYTHONPATH = "src"
python -m crowd_tracker.train_detector_cli `
  --data configs\visdrone_person.example.yaml `
  --epochs 50 --imgsz 640 --batch 4 --device 0 `
  --project runs\detect --name visdrone_person_50e
```

Use the resulting `best.pt` only after reviewing validation metrics and qualitative false positives/negatives. Point a local, ignored runtime configuration at the selected checkpoint for an integration run.
