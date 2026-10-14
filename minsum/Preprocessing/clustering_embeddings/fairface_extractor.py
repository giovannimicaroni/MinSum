"""FairFace penultimate-layer extractor for clustering embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from minsum.Core.embedding_extraction import EmbeddingExtractor, ExtractionConfig


@dataclass
class FairFaceConfig(ExtractionConfig):
    weights_path: Path | None = None
    device: str | None = None
    num_outputs: int = 18
    expected_dim: int = 512


def clean_state_dict_keys(state_dict):
    return {key.replace("module.", ""): value for key, value in state_dict.items()}


class FairFacePenultimateExtractor(EmbeddingExtractor):
    """Extract features from the penultimate layer of FairFace ResNet34."""

    def __init__(self, config: FairFaceConfig) -> None:
        super().__init__(config)
        self.model = None
        self.transform = None
        self.device = None

    @property
    def fairface_config(self) -> FairFaceConfig:
        return self.config  # type: ignore[return-value]

    def _load_model(self):
        if self.model is not None:
            return self.model

        import torch
        import torch.nn as nn
        from torchvision import models, transforms

        cfg = self.fairface_config
        self.device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = models.resnet34()
        model.fc = nn.Linear(model.fc.in_features, cfg.num_outputs)
        if cfg.weights_path is not None:
            state_dict = torch.load(str(cfg.weights_path), map_location=self.device)
            model.load_state_dict(clean_state_dict_keys(state_dict), strict=False)
        feature_model = nn.Sequential(*(list(model.children())[:-1]))
        feature_model.to(self.device)
        feature_model.eval()
        self.model = feature_model
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        return self.model

    def extract_one(self, image_path: Path) -> np.ndarray:
        import torch
        from PIL import Image

        model = self._load_model()
        assert self.transform is not None
        assert self.device is not None

        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = model(tensor).reshape(-1).detach().cpu().numpy().astype(np.float32)
        expected_dim = self.fairface_config.expected_dim
        if expected_dim and features.shape[0] != expected_dim:
            raise ValueError(f"Expected feature dim {expected_dim}, got {features.shape[0]}")
        return features
