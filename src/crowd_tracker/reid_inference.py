"""Inference adapter for a trained AG-ReID appearance encoder."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet50_Weights

from .reid_training import SiameseReIdEncoder


class TrainedReIdEncoder:
    """Loads a checkpoint and returns L2-normalized vectors from BGR crops."""

    def __init__(self, checkpoint: str | Path, device: str = "auto", image_size: int = 192) -> None:
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Re-ID checkpoint is missing: {self.checkpoint}")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        payload = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        self.backbone = str(payload["backbone"])
        self.embedding_dim = int(payload["embedding_dim"])
        self.model = SiameseReIdEncoder(self.backbone, self.embedding_dim, pretrained=False).to(self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        weights = ResNet50_Weights.IMAGENET1K_V2.transforms()
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=weights.mean, std=weights.std),
        ])

    @torch.inference_mode()
    def embed_bgr(self, crop: np.ndarray) -> np.ndarray:
        return self.embed_many_bgr([crop])[0]

    @torch.inference_mode()
    def embed_many_bgr(self, crops: list[np.ndarray]) -> np.ndarray:
        """Encode a batch for benchmark throughput while retaining crop API use."""
        if not crops:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        tensors = [self.transform(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))) for crop in crops]
        return self.model(torch.stack(tensors).to(self.device)).detach().cpu().numpy().astype(np.float32)

    @property
    def method_label(self) -> str:
        return f"AG-ReID trained {self.backbone} triplet encoder"
