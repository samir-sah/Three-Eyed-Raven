"""Fine-tune a shared Re-ID encoder with triplet loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .reid_training import SiameseReIdEncoder, TripletManifestDataset, load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Siamese-style Re-ID encoder using triplet loss.")
    parser.add_argument("--manifest", required=True, help="JSONL image manifest with path, person_id, camera_id, split.")
    parser.add_argument("--root", default=".", help="Directory used to resolve manifest image paths.")
    parser.add_argument("--output", default="models/reid_encoder.pt")
    parser.add_argument("--backbone", choices=["resnet18", "resnet50"], default="resnet50")
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-pretrained", action="store_true", help="Avoid downloading ImageNet backbone weights.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.embedding_dim < 8:
        raise ValueError("epochs >= 1, batch-size >= 2, and embedding-dim >= 8 are required.")

    dataset = TripletManifestDataset(load_manifest(args.manifest, split="train"), root=args.root, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=args.device.startswith("cuda"))
    device = torch.device(args.device)
    model = SiameseReIdEncoder(args.backbone, args.embedding_dim, pretrained=not args.no_pretrained).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.TripletMarginLoss(margin=args.margin, p=2)
    losses = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = batches = 0
        for anchor, positive, negative in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(anchor.to(device)), model(positive.to(device)), model(negative.to(device)))
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        mean_loss = total / max(1, batches)
        losses.append(mean_loss)
        print(json.dumps({"epoch": epoch, "triplet_loss": round(mean_loss, 6)}))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "backbone": args.backbone, "embedding_dim": args.embedding_dim, "losses": losses}, destination)
    print(json.dumps({"checkpoint": str(destination), "final_triplet_loss": losses[-1], "device": str(device)}, indent=2))


if __name__ == "__main__":
    main()
