"""TransFace verification embeddings using official ViT checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from minsum.Core.embedding_extraction import EmbeddingExtractor, ExtractionConfig


@dataclass
class TransFaceConfig(ExtractionConfig):
    weights_path: Path | None = None
    network: str = "vit_s_dp005_mask_0"
    device: str | None = None
    image_size: tuple[int, int] = (112, 112)
    expected_dim: int = 512
    l2_normalize: bool = True


class TransFaceExtractor(EmbeddingExtractor):
    """Extract TransFace embeddings from aligned 112x112 face crops."""

    def __init__(self, config: TransFaceConfig | None = None) -> None:
        super().__init__(config or TransFaceConfig())
        self.model = None
        self.device = None

    @property
    def transface_config(self) -> TransFaceConfig:
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
            raise ValueError("TransFace checkpoint must contain a state dictionary")
        for key in ("state_dict", "model_state_dict", "backbone", "net", "model"):
            state_dict = checkpoint.get(key)
            if isinstance(state_dict, dict):
                break
        else:
            state_dict = checkpoint
        if not isinstance(state_dict, dict):
            raise ValueError("TransFace checkpoint state dictionary is not a dictionary")
        return {
            key: value
            for key, value in state_dict.items()
            if isinstance(key, str)
        }

    @staticmethod
    def _strip_prefixes(key: str, prefixes: tuple[str, ...]) -> str:
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key.removeprefix(prefix)
                    changed = True
        return key

    @classmethod
    def _candidate_state_dicts(cls, state_dict: dict[str, object]) -> list[dict[str, object]]:
        prefix_sets = [
            tuple(),
            ("module.",),
            ("model.",),
            ("backbone.",),
            ("net.",),
            ("module.", "model.", "backbone.", "net."),
        ]
        candidates = []
        seen_signatures = set()
        for prefixes in prefix_sets:
            candidate = {
                cls._strip_prefixes(key, prefixes): value
                for key, value in state_dict.items()
            }
            signature = tuple(sorted(candidate.keys())[:20])
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                candidates.append(candidate)
        return candidates

    @classmethod
    def _select_compatible_state_dict(
        cls,
        state_dict: dict[str, object],
        model_state_dict: dict[str, object],
    ) -> dict[str, object]:
        best_candidate = None
        best_key_matches = -1
        best_shape_matches = -1
        for candidate in cls._candidate_state_dicts(state_dict):
            key_matches = set(candidate).intersection(model_state_dict)
            shape_matches = 0
            for key in key_matches:
                value = candidate[key]
                expected = model_state_dict[key]
                if hasattr(value, "shape") and hasattr(expected, "shape"):
                    shape_matches += tuple(value.shape) == tuple(expected.shape)
                else:
                    shape_matches += 1
            if (len(key_matches), shape_matches) > (best_key_matches, best_shape_matches):
                best_candidate = candidate
                best_key_matches = len(key_matches)
                best_shape_matches = shape_matches

        if best_candidate is None or best_key_matches == 0:
            sample_keys = list(state_dict)[:20]
            raise ValueError(
                "Could not match TransFace checkpoint keys to the model. "
                f"Sample checkpoint keys: {sample_keys}"
            )

        cleaned = {}
        mismatched = []
        for key, value in best_candidate.items():
            expected = model_state_dict.get(key)
            if expected is None:
                continue
            if hasattr(value, "shape") and hasattr(expected, "shape"):
                if tuple(value.shape) != tuple(expected.shape):
                    mismatched.append((key, tuple(value.shape), tuple(expected.shape)))
                    continue
            cleaned[key] = value
        if mismatched:
            raise ValueError(
                "TransFace checkpoint has incompatible tensor shapes. "
                f"Examples: {mismatched[:10]}"
            )
        return cleaned

    def _load_model(self):
        if self.model is not None:
            return self.model

        import torch

        from .vit import build_transface

        cfg = self.transface_config
        if cfg.weights_path is None:
            raise ValueError("TransFace weights_path is required")
        weights_path = Path(cfg.weights_path).expanduser()
        if not weights_path.is_file():
            raise FileNotFoundError(f"TransFace weights not found: {weights_path}")

        self.device = self._resolve_device(cfg.device)
        model = build_transface(cfg.network, num_features=cfg.expected_dim)
        try:
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(weights_path, map_location="cpu")
        raw_state_dict = self._checkpoint_state_dict(checkpoint)
        state_dict = self._select_compatible_state_dict(raw_state_dict, model.state_dict())
        load_result = model.load_state_dict(state_dict, strict=False)
        missing = set(load_result.missing_keys)
        unexpected = set(load_result.unexpected_keys)
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint {weights_path} is incompatible with network "
                f"{cfg.network!r}: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        self.model = model.to(self.device).eval()
        return self.model

    def _to_input(self, image: np.ndarray):
        import cv2
        import torch

        cfg = self.transface_config
        resized = cv2.resize(image, cfg.image_size, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        normalized = (normalized - 0.5) / 0.5
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
            output = model(self._to_input(image).to(self.device))
        embedding = output[0] if isinstance(output, tuple) else output
        result = embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if result.ndim != 1:
            raise ValueError(f"Expected 1D embedding, got shape {result.shape}")
        expected_dim = self.transface_config.expected_dim
        if expected_dim and result.shape[0] != expected_dim:
            raise ValueError(f"Expected embedding dim {expected_dim}, got {result.shape[0]}")
        if self.transface_config.l2_normalize:
            norm = np.linalg.norm(result)
            if norm == 0:
                raise ValueError("Embedding norm is zero")
            result = result / norm
        return result.astype(np.float32)
