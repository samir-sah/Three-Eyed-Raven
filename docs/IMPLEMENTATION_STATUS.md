# Implementation status

Last verified: 30 September 2026

## Delivered and verified

| Roadmap area | Implemented evidence | Status |
| --- | --- | --- |
| Environment and reproducibility | Python virtual environment, CUDA PyTorch, pinned requirements, tests, Docker compose, README and dataset instructions | Complete locally |
| Video ingestion and preprocessing | File/RTSP sources, frame preprocessing, reconnect settings, bounded threaded capture, timestamp pairing primitives | Code complete; live field validation pending an authorised camera |
| Detection | YOLO person detection; VisDrone DET train/validation preparation and fine-tuning entry point | Baseline and training workflow verified |
| Local multi-object tracking | ByteTrack per camera, annotated video, JSONL observations, local-ID safeguards | Complete baseline |
| MOT evaluation | MOT ground-truth import, direct image-sequence run, MOTA/MOTP/ID-switch metrics | Complete baseline |
| Cross-camera review | Explainable HSV appearance ranking, candidate crops, trained Siamese Re-ID training/evaluation infrastructure | Review-only baseline complete; trained model awaits licensed data |
| Dashboard and persistence | Next.js dashboard, playback, alerts, profiling, benchmark cards; FastAPI and SQLite run index | Complete local demo |
| Integration and operations | Multiple recorded streams, latency profiling, source preflight, API-key option, container definitions | Complete local demo; deployment validation pending |

## Measured baselines

These figures are reproducible baselines, **not final project claims**.

| Experiment | Data and configuration | Result |
| --- | --- | --- |
| Tracking | MOT17-02-FRCNN, first 120 frames, COCO-pretrained YOLO11n + ByteTrack | 16.95 processing FPS; precision 0.842; recall 0.206; F1 0.331; MOTA 0.167; MOTP 0.805; 2 ID switches |
| Detector smoke training | VisDrone DET person labels, YOLO11n, 640 px, batch 4, 1 epoch on RTX 3050 Laptop GPU | mAP50 0.204; mAP50-95 0.062; checkpoint saved to `runs/detect/outputs/detection/visdrone_person_smoke/weights/best.pt` |

The detector smoke run established that CUDA, labels, validation, logging, and checkpoint export work. A one-epoch model is not suitable for deployment; run the documented multi-epoch experiment before comparing it with the baseline.

## Remaining work that cannot be truthfully marked complete yet

1. **Full detector experiment.** Train/validate for an agreed schedule (for example 50 epochs), preserve the run directory, and compare the best checkpoint against the COCO baseline on a held-out aerial set.
2. **Trained cross-camera Re-ID.** PRAI-1581, CARGO, and AG-ReID require their respective access/licence conditions. Obtain them legally, prepare a manifest, train the supplied Siamese model, then report Rank-1/5/10 and mAP. The application must keep matching as operator review, never automatic identity confirmation.
3. **MOT17 full benchmark.** Evaluate all agreed sequences and detector variants, then record aggregate MOTA, MOTP, IDF1, HOTA, ID switches, and latency. The current 120-frame run is intentionally a small sanity benchmark.
4. **Authorised live field validation.** Calibrate zones, source credentials, retention policy, alert thresholds, permission notices, and test with real drone/CCTV feeds only after approval.
5. **Academic finalisation.** Replace the baseline values in the report/slides with results from the complete experiments, cite data licences, and rehearse the live demo fallback.

## Exact command for the next experiment

```powershell
$env:PYTHONPATH = "src"
python -m crowd_tracker.train_detector_cli `
  --data configs\visdrone_person.example.yaml `
  --epochs 50 --imgsz 640 --batch 4 --device 0 `
  --project runs\detect --name visdrone_person_50e
```

Use the resulting `best.pt` only after reviewing validation metrics and qualitative false positives/negatives. Point a local, ignored runtime configuration at the selected checkpoint for an integration run.
