"""Trainable shared-backbone person Re-ID model and manifest-backed triplets."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50


@dataclass(frozen=True)
class ReIdSample:
    path: str
    person_id: str
    camera_id: str
    split: str


def load_manifest(path: str | Path, split: str | None = None) -> list[ReIdSample]:
    source = Path(path)
    records: list[ReIdSample] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            sample = ReIdSample(
                path=str(raw["path"]),
                person_id=str(raw["person_id"]),
                camera_id=str(raw["camera_id"]),
                split=str(raw.get("split", "train")),
            )
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid Re-ID manifest line {line_number}.") from error
        if split is None or sample.split == split:
            records.append(sample)
    if not records:
        raise ValueError(f"No Re-ID samples found for split '{split}' in {source}.")
    return records


class TripletManifestDataset(Dataset):
    """Randomly samples anchor/positive/negative images from a labelled manifest."""

    def __init__(self, samples: list[ReIdSample], root: str | Path = ".", image_size: int = 256, seed: int = 42) -> None:
        self.samples = samples
        self.root = Path(root)
        self.random = random.Random(seed)
        self.by_id: dict[str, list[ReIdSample]] = {}
        for sample in samples:
            self.by_id.setdefault(sample.person_id, []).append(sample)
        self.eligible = [sample for sample in samples if len(self.by_id[sample.person_id]) >= 2 and len(self.by_id) >= 2]
        if not self.eligible:
            raise ValueError("Triplet training needs at least two identities and two samples for at least one identity.")
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=ResNet50_Weights.IMAGENET1K_V2.transforms().mean, std=ResNet50_Weights.IMAGENET1K_V2.transforms().std),
        ])

    def __len__(self) -> int:
        return len(self.eligible)

    def __getitem__(self, index: int):
        anchor = self.eligible[index]
        positive_options = [sample for sample in self.by_id[anchor.person_id] if sample.path != anchor.path]
        positive = self.random.choice(positive_options)
        negative_id = self.random.choice([identity for identity in self.by_id if identity != anchor.person_id])
        negative = self.random.choice(self.by_id[negative_id])
        return tuple(self._read(sample.path) for sample in (anchor, positive, negative))

    def _read(self, relative_path: str) -> torch.Tensor:
        with Image.open(self.root / relative_path) as image:
            return self.transform(image.convert("RGB"))


class SiameseReIdEncoder(nn.Module):
    """Shared ResNet encoder producing L2-normalized appearance embeddings."""

    def __init__(self, backbone: str = "resnet50", embedding_dim: int = 512, pretrained: bool = True) -> None:
        super().__init__()
        if backbone == "resnet50":
            network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        elif backbone == "resnet18":
            network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError("backbone must be resnet18 or resnet50.")
        features = network.fc.in_features
        network.fc = nn.Identity()
        self.backbone = network
        self.projector = nn.Sequential(nn.Linear(features, embedding_dim), nn.BatchNorm1d(embedding_dim))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.projector(self.backbone(images)), p=2, dim=1)
