from __future__ import annotations

import numpy as np
import torch
from torchvision.models import resnet50, ResNet50_Weights


class FrozenResNet50Embedder:
    """Frozen ImageNet ResNet-50 feature extractor.

    Expects a batch tensor shaped [N,3,224,224] already scaled to [0,1].
    Standard ImageNet preprocessing is applied through the torchvision weights.
    """

    def __init__(self, device=None):
        self.weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=self.weights)
        model.fc = torch.nn.Identity()
        model.eval()
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[None,:,None,None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[None,:,None,None]

    @torch.inference_mode()
    def encode(self, batch):
        if isinstance(batch, np.ndarray):
            batch = torch.from_numpy(batch)
        x = batch.to(self.device, dtype=torch.float32)
        if x.ndim != 4 or x.shape[1:] != (3, 224, 224):
            raise ValueError("expected [N,3,224,224]")
        x = (x - self.mean) / self.std
        z = self.model(x)
        return z.detach().cpu().numpy()


def average_triplanar_embeddings(axial, coronal, sagittal, embedder):
    """Average three 2048-D embeddings as in the manuscript."""
    batch = np.stack([axial, coronal, sagittal], axis=0)
    z = embedder.encode(batch)
    return z.mean(axis=0)
