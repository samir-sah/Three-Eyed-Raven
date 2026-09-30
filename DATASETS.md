# Dataset Setup and Research Data Status

Raw datasets stay under the gitignored `data/` directory. Do not commit raw images, annotations, checkpoints, or any camera credentials.

| Dataset | Roadmap use | Access status | Project support |
|---|---|---|---|
| VisDrone2019 DET/MOT | Aerial detector and tracker evaluation | Official public download; large files | DET-to-YOLO preparation CLI |
| MOT17 | Ground tracking benchmark | Official public download; ~5.86 GB archive | MOT-to-JSONL import and tracking evaluator |
| PRAI-1581 | Aerial-ground Re-ID training | Obtain from its authors/licence terms | Re-ID manifest evaluator ready |
| CARGO | Aerial-ground Re-ID training | Obtain from its authors/licence terms | Re-ID manifest evaluator ready |
| AG-ReID | Aerial-ground Re-ID training/evaluation | Obtain from its authors/licence terms | Re-ID manifest evaluator ready |

## VisDrone detector preparation

After downloading the official VisDrone DET train and validation archives, prepare each split. The default maps VisDrone category 1 (`pedestrian`) and 2 (`people`) into the project’s single `person` class.

```bat
set PYTHONPATH=src
python -m crowd_tracker.visdrone_prepare_cli --images data\raw\VisDrone2019-DET-train\images --annotations data\raw\VisDrone2019-DET-train\annotations --output data\processed\visdrone_person\train
python -m crowd_tracker.visdrone_prepare_cli --images data\raw\VisDrone2019-DET-val\images --annotations data\raw\VisDrone2019-DET-val\annotations --output data\processed\visdrone_person\val
python -m crowd_tracker.train_detector_cli --data configs\visdrone_person.yaml --epochs 50 --imgsz 960 --device 0
```

Copy `configs/visdrone_person.example.yaml` to `configs/visdrone_person.yaml` and adjust `path` first.

## Benchmark integrity

Keep training, validation, and test sequences separated. Store a copy of each generated metric report with the checkpoint name, GPU, input size, and command-line arguments. Do not report demo-video values as benchmark results.

## MOT17 ByteTrack benchmark run

After extracting the official MOT17 archive, run an image sequence and export the identical frame range from its ground truth. This produces a real, reproducible baseline—although COCO-pretrained YOLO has not yet been fine-tuned for MOT17, so treat the first result as a baseline rather than a target claim.

```bat
set PYTHONPATH=src
python -m crowd_tracker.mot_run_cli --images data\raw\MOT17\train\MOT17-02-FRCNN\img1 --output artifacts\mot17_02_baseline --max-frames 120 --device auto
python -m crowd_tracker.mot_import_cli --input data\raw\MOT17\train\MOT17-02-FRCNN\gt\gt.txt --output artifacts\mot17_02_ground_truth.jsonl --end-frame 120
python -m crowd_tracker.tracking_eval_cli --ground-truth artifacts\mot17_02_ground_truth.jsonl --predictions artifacts\mot17_02_baseline\observations.jsonl --output artifacts\mot17_02_baseline\evaluation.json
```

## MOT17 batch benchmark

After selecting a trained detector checkpoint, evaluate the available FRCNN sequences in one repeatable command. The output contains per-sequence artifacts plus `benchmark_summary.json` with aggregate precision, recall, F1, MOTA, MOTP, IDF1, HOTA, ID switches, and latency from each sequence summary.

```bat
set PYTHONPATH=src
python -m crowd_tracker.mot_benchmark_cli --mot-root data\raw\MOT17\train --output artifacts\mot17_visdrone_person_50e --model runs\detect\runs\detect\visdrone_person_50e\weights\best.pt --device 0
```

For a quick smoke benchmark before the full run, add `--max-frames 120`. Keep its report separate from the full-sequence result.

## Resuming memory-constrained detector training

If a machine exhausts memory in a data-loader worker, resume from the run's `last.pt` checkpoint with fewer workers. This preserves the experiment's saved epoch state.

```bat
set PYTHONPATH=src
python -m crowd_tracker.train_detector_cli --data configs\visdrone_person.example.yaml --resume runs\detect\runs\detect\visdrone_person_50e\weights\last.pt --workers 1
```

## Re-ID training preparation

Create a JSONL manifest with `path`, `person_id`, `camera_id`, and `split`; use `configs/reid_train.example.jsonl` as the schema. Each train identity needs at least two images, ideally from different camera viewpoints. Then train a shared ResNet encoder using triplet loss:

```bat
set PYTHONPATH=src
python -m crowd_tracker.reid_train_cli --manifest configs\reid_train.jsonl --root data\processed\ag_reid --output models\reid_encoder.pt --backbone resnet50 --device cuda
```

The training checkpoint can then be used to export query and gallery embeddings for `crowd_tracker.reid_eval_cli`. Do not run or present this as an experiment until the dataset licence/terms allow local training and the manifest has disjoint train/test identities.
