"""AdaFace verification embeddings using official checkpoint-compatible backbones."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import types

import numpy as np

from minsum.Core.embedding_extraction import EmbeddingExtractor, ExtractionConfig


@dataclass
class AdaFaceConfig(ExtractionConfig):
    weights_path: Path | None = None
    architecture: str = "ir_50"
    device: str | None = None
    image_size: tuple[int, int] = (112, 112)
    expected_dim: int = 512


class AdaFaceExtractor(EmbeddingExtractor):
    """Extract L2-normalized AdaFace embeddings from aligned face crops."""

    def __init__(self, config: AdaFaceConfig | None = None) -> None:
        super().__init__(config or AdaFaceConfig())
        self.model = None
        self.device = None

    @property
    def adaface_config(self) -> AdaFaceConfig:
        return self.config  # type: ignore[return-value]

    @staticmethod
    def _resolve_device(requested_device: str | None):
        import torch

        if requested_device is not None:
            device = torch.device(requested_device)
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available")
            if device.type == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("MPS was requested but is not available")
            return device
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _checkpoint_state_dict(checkpoint: object) -> dict[str, object]:
        if not isinstance(checkpoint, dict):
            raise ValueError("AdaFace checkpoint must contain a state dictionary")
        state_dict = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state_dict, dict):
            raise ValueError("AdaFace checkpoint 'state_dict' is not a dictionary")

        official = {
            key.removeprefix("model."): value
            for key, value in state_dict.items()
            if isinstance(key, str) and key.startswith("model.")
        }
        if official:
            return official
        return {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
            if isinstance(key, str)
        }

    @staticmethod
    def _install_lightning_checkpoint_stub() -> None:
        """Let trusted Lightning checkpoints unpickle without installing Lightning."""

        if "pytorch_lightning.callbacks.model_checkpoint" in sys.modules:
            return

        pl_module = sys.modules.setdefault(
            "pytorch_lightning",
            types.ModuleType("pytorch_lightning"),
        )
        callbacks_module = sys.modules.setdefault(
            "pytorch_lightning.callbacks",
            types.ModuleType("pytorch_lightning.callbacks"),
        )
        model_checkpoint_module = types.ModuleType(
            "pytorch_lightning.callbacks.model_checkpoint"
        )

        class ModelCheckpoint:
            pass

        model_checkpoint_module.ModelCheckpoint = ModelCheckpoint
        callbacks_module.model_checkpoint = model_checkpoint_module
        pl_module.callbacks = callbacks_module
        sys.modules[
            "pytorch_lightning.callbacks.model_checkpoint"
        ] = model_checkpoint_module

    def _load_model(self):
        if self.model is not None:
            return self.model

        import torch

        from .backbone import build_model

        cfg = self.adaface_config
        if cfg.weights_path is None:
            raise ValueError("AdaFace weights_path is required")
        weights_path = Path(cfg.weights_path).expanduser()
        if not weights_path.is_file():
            raise FileNotFoundError(f"AdaFace weights not found: {weights_path}")

        self.device = self._resolve_device(cfg.device)
        model = build_model(cfg.architecture)
        try:
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(weights_path, map_location="cpu")
        except Exception as exc:
            if "Weights only load failed" not in str(exc):
                raise
            self._install_lightning_checkpoint_stub()
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = self._checkpoint_state_dict(checkpoint)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Checkpoint {weights_path} is incompatible with architecture "
                f"{cfg.architecture!r}: {exc}"
            ) from exc
        self.model = model.to(self.device).eval()
        return self.model

    def _to_input(self, image: np.ndarray):
        import cv2
        import torch

        cfg = self.adaface_config
        resized = cv2.resize(image, cfg.image_size, interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 127.5 - 1.0
        chw = np.ascontiguousarray(normalized.transpose(2, 0, 1))
        return torch.from_numpy(chw).unsqueeze(0)

    def extract_one(self, image_path: Path) -> np.ndarray:
        import cv2
        import torch

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("cv2.imread returned None")

        model = self._load_model()
        with torch.inference_mode():
            embedding, _ = model(self._to_input(image).to(self.device))
        result = embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if result.ndim != 1:
            raise ValueError(f"Expected 1D embedding, got shape {result.shape}")
        expected_dim = self.adaface_config.expected_dim
        if expected_dim and result.shape[0] != expected_dim:
            raise ValueError(f"Expected embedding dim {expected_dim}, got {result.shape[0]}")
        norm = np.linalg.norm(result)
        if norm == 0:
            raise ValueError("Embedding norm is zero")
        return (result / norm).astype(np.float32)
