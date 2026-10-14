"""InsightFace ArcFace extractor for verification embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from minsum.Core.embedding_extraction import EmbeddingExtractor, ExtractionConfig


@dataclass
class InsightFaceConfig(ExtractionConfig):
    model_name: str = "buffalo_l"
    ctx_id: int = 0
    det_size: tuple[int, int] = (640, 640)
    expected_dim: int = 512
    mode: str = "detect"
    model_root: Path | None = None
    recognition_model: str = "w600k_r50.onnx"
    image_size: tuple[int, int] = (112, 112)


class InsightFaceArcFaceExtractor(EmbeddingExtractor):
    """Extract normed ArcFace embeddings with InsightFace."""

    def __init__(self, config: InsightFaceConfig | None = None) -> None:
        super().__init__(config or InsightFaceConfig())
        self.app = None
        self.recognition_model = None

    @property
    def insightface_config(self) -> InsightFaceConfig:
        return self.config  # type: ignore[return-value]

    def _load_app(self):
        if self.app is None:
            from insightface.app import FaceAnalysis

            cfg = self.insightface_config
            self.app = FaceAnalysis(name=cfg.model_name)
            self.app.prepare(ctx_id=cfg.ctx_id, det_size=cfg.det_size)
        return self.app

    def _recognition_model_path(self) -> Path:
        cfg = self.insightface_config
        if cfg.model_root is not None:
            model_root = Path(cfg.model_root).expanduser()
        else:
            model_root = Path.home() / ".insightface" / "models" / cfg.model_name
        return model_root / cfg.recognition_model

    def _load_recognition_model(self):
        if self.recognition_model is None:
            from insightface.model_zoo import get_model

            cfg = self.insightface_config
            model_path = self._recognition_model_path()
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Recognition model not found at {model_path}. "
                    "Run detect mode once to download the InsightFace model pack, "
                    "or pass --model-root /path/to/.insightface/models/buffalo_l."
                )
            self.recognition_model = get_model(str(model_path))
            self.recognition_model.prepare(ctx_id=cfg.ctx_id)
        return self.recognition_model

    @staticmethod
    def choose_best_face(faces):
        if len(faces) == 0:
            raise ValueError("No face detected")

        def area(face) -> float:
            x1, y1, x2, y2 = face.bbox
            return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))

        return max(faces, key=area)

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        embedding = np.asarray(embedding, dtype=np.float32).squeeze()
        if embedding.ndim != 1:
            raise ValueError(f"Expected 1D embedding, got shape {embedding.shape}")
        norm = np.linalg.norm(embedding)
        if norm == 0:
            raise ValueError("Embedding norm is zero")
        return (embedding / norm).astype(np.float32)

    def _extract_with_detection(self, image) -> np.ndarray:
        faces = self._load_app().get(image)
        face = self.choose_best_face(faces)
        return self._normalize(face.normed_embedding)

    def _extract_prealigned(self, image) -> np.ndarray:
        import cv2

        cfg = self.insightface_config
        resized = cv2.resize(image, cfg.image_size)
        embedding = self._load_recognition_model().get_feat(resized)
        return self._normalize(embedding)

    def _validate_embedding(self, embedding: np.ndarray) -> np.ndarray:
        expected_dim = self.insightface_config.expected_dim
        if expected_dim and embedding.shape[0] != expected_dim:
            raise ValueError(f"Expected embedding dim {expected_dim}, got {embedding.shape[0]}")
        return embedding.astype(np.float32)

    def extract_one(self, image_path: Path) -> np.ndarray:
        import cv2

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("cv2.imread returned None")

        mode = self.insightface_config.mode
        if mode == "detect":
            embedding = self._extract_with_detection(image)
        elif mode == "prealigned":
            embedding = self._extract_prealigned(image)
        else:
            raise ValueError(f"Unsupported InsightFace extraction mode: {mode}")
        return self._validate_embedding(embedding)
