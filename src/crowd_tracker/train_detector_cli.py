"""Reproducible Ultralytics fine-tuning entry point for the aerial detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a YOLO model on a prepared person dataset.")
    parser.add_argument("--data", required=True, help="YOLO dataset YAML path.")
    parser.add_argument("--model", default="yolo11n.pt", help="Starting pretrained Ultralytics checkpoint.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=960, help="Use a larger size for small aerial people when GPU memory permits.")
    parser.add_argument("--batch", type=int, default=-1, help="-1 lets Ultralytics choose the batch size.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4, help="Data-loader worker count; reduce this on memory-constrained systems.")
    parser.add_argument("--resume", help="Resume an interrupted Ultralytics run from its last.pt checkpoint.")
    parser.add_argument("--project", default="outputs/detection")
    parser.add_argument("--name", default="visdrone_person")
    args = parser.parse_args()
    if args.epochs < 1 or args.imgsz < 32 or args.workers < 0:
        raise ValueError("epochs must be >= 1 and imgsz must be >= 32.")
    if args.resume and not Path(args.resume).is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
    if not args.resume and not Path(args.data).is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")

    from ultralytics import YOLO

    model = YOLO(args.resume or args.model)
    if args.resume:
        results = model.train(resume=True, workers=args.workers)
    else:
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=args.project,
            name=args.name,
            exist_ok=True,
            pretrained=True,
        )
    print(json.dumps({"project": args.project, "name": args.name, "results": str(results)}, indent=2))


if __name__ == "__main__":
    main()
