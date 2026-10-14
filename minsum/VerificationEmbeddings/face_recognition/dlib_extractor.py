"""Verification embeddings from the dlib/face_recognition model family."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path

import numpy as np

from minsum.Core.embedding_extraction import EmbeddingExtractor, ExtractionConfig


@dataclass
class FaceRecognitionConfig(ExtractionConfig):
    mode: str = "prealigned"
    detector_model: str = "hog"
    landmark_model: str = "small"
    num_jitters: int = 1
    normalize: bool = True
    expected_dim: int = 128


class FaceRecognitionExtractor(EmbeddingExtractor):
    """Extract L2-normalized 128D dlib face embeddings."""

    def __init__(self, config: FaceRecognitionConfig | None = None) -> None:
        super().__init__(config or FaceRecognitionConfig())
        self.hog_detector = None
        self.cnn_detector = None
        self.shape_predictor = None
        self.recognition_model = None

    @property
    def fr_config(self) -> FaceRecognitionConfig:
        return self.config  # type: ignore[return-value]

    @staticmethod
    def _models_dir() -> Path:
        spec = importlib.util.find_spec("face_recognition_models")
        if spec is None or spec.origin is None:
            raise ImportError(
                "face_recognition_models is required for dlib model files. "
                "Install the conda package face_recognition_models."
            )
        return Path(spec.origin).parent / "models"

    def _model_path(self, filename: str) -> Path:
        path = self._models_dir() / filename
        if not path.exists():
            raise FileNotFoundError(f"dlib model file not found: {path}")
        return path

    def _load_shape_predictor(self):
        if self.shape_predictor is None:
            import dlib

            filename = (
                "shape_predictor_5_face_landmarks.dat"
                if self.fr_config.landmark_model == "small"
                else "shape_predictor_68_face_landmarks.dat"
            )
            self.shape_predictor = dlib.shape_predictor(str(self._model_path(filename)))
        return self.shape_predictor

    def _load_recognition_model(self):
        if self.recognition_model is None:
            import dlib

            self.recognition_model = dlib.face_recognition_model_v1(
                str(self._model_path("dlib_face_recognition_resnet_model_v1.dat"))
            )
        return self.recognition_model

    def _load_detector(self):
        import dlib

        if self.fr_config.detector_model == "hog":
            if self.hog_detector is None:
                self.hog_detector = dlib.get_frontal_face_detector()
            return self.hog_detector
        if self.cnn_detector is None:
            self.cnn_detector = dlib.cnn_face_detection_model_v1(
                str(self._model_path("mmod_human_face_detector.dat"))
            )
        return self.cnn_detector

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        embedding = np.asarray(embedding, dtype=np.float32).squeeze()
        if embedding.ndim != 1:
            raise ValueError(f"Expected 1D embedding, got shape {embedding.shape}")
        norm = np.linalg.norm(embedding)
        if norm == 0:
            raise ValueError("Embedding norm is zero")
        return (embedding / norm).astype(np.float32)

    @staticmethod
    def _largest_rect(rectangles):
        if len(rectangles) == 0:
            raise ValueError("No face detected")

        def as_rect(candidate):
            return getattr(candidate, "rect", candidate)

        def area(candidate) -> int:
            rect = as_rect(candidate)
            return max(0, rect.right() - rect.left()) * max(0, rect.bottom() - rect.top())

        return as_rect(max(rectangles, key=area))

    @staticmethod
    def _whole_image_rect(image: np.ndarray):
        import dlib

        height, width = image.shape[:2]
        return dlib.rectangle(left=0, top=0, right=width - 1, bottom=height - 1)

    def _face_rect(self, image: np.ndarray):
        cfg = self.fr_config
        if cfg.mode == "prealigned":
            return self._whole_image_rect(image)
        if cfg.mode == "detect":
            rectangles = self._load_detector()(image, 1)
            return self._largest_rect(rectangles)
        raise ValueError(f"Unsupported face_recognition mode: {cfg.mode}")

    def _validate_embedding(self, embedding: np.ndarray) -> np.ndarray:
        expected_dim = self.fr_config.expected_dim
        if expected_dim and embedding.shape[0] != expected_dim:
            raise ValueError(f"Expected embedding dim {expected_dim}, got {embedding.shape[0]}")
        return embedding.astype(np.float32)

    def extract_one(self, image_path: Path) -> np.ndarray:
        from PIL import Image

        cfg = self.fr_config
        image = np.asarray(Image.open(image_path).convert("RGB"))
        rect = self._face_rect(image)
        shape = self._load_shape_predictor()(image, rect)
        descriptor = self._load_recognition_model().compute_face_descriptor(
            image,
            shape,
            cfg.num_jitters,
        )
        embedding = np.asarray(descriptor, dtype=np.float32)
        if cfg.normalize:
            embedding = self._normalize(embedding)
        return self._validate_embedding(embedding)
