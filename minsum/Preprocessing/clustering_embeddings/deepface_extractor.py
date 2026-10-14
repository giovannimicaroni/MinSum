"""DeepFace embedding extractor for clustering embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from minsum.Core.embedding_extraction import EmbeddingExtractor, ExtractionConfig


@dataclass
class DeepFaceConfig(ExtractionConfig):
    model_name: str = "ArcFace"
    detector_backend: str = "skip"
    align: bool = False
    enforce_detection: bool = False
    normalization: str = "base"


class DeepFaceEmbeddingExtractor(EmbeddingExtractor):
    """Extract embeddings with DeepFace.represent."""

    def __init__(self, config: DeepFaceConfig | None = None) -> None:
        super().__init__(config or DeepFaceConfig())
        self.deepface = None

    @property
    def deepface_config(self) -> DeepFaceConfig:
        return self.config  # type: ignore[return-value]

    def _load_deepface(self):
        if self.deepface is None:
            from deepface import DeepFace

            self.deepface = DeepFace
        return self.deepface

    def extract_one(self, image_path: Path) -> np.ndarray:
        cfg = self.deepface_config
        result = self._load_deepface().represent(
            img_path=str(image_path),
            model_name=cfg.model_name,
            detector_backend=cfg.detector_backend,
            enforce_detection=cfg.enforce_detection,
            align=cfg.align,
            normalization=cfg.normalization,
        )
        if isinstance(result, list):
            if not result:
                raise ValueError("DeepFace returned no faces")
            embedding = result[0].get("embedding")
        elif isinstance(result, dict):
            embedding = result.get("embedding")
        else:
            raise TypeError(f"Unexpected DeepFace result type: {type(result)}")
        if embedding is None:
            raise ValueError("DeepFace result did not contain an embedding")
        return np.asarray(embedding, dtype=np.float32)
