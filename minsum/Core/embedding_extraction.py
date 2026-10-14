"""Base classes for embedding extractors."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from minsum.Core.artifact_store import ensure_dir, save_dataframe


@dataclass
class ExtractionResult:
    split: str
    embeddings: np.ndarray
    metadata: pd.DataFrame
    errors: pd.DataFrame


@dataclass
class ExtractionConfig:
    checkpoint_every: int = 0
    allow_empty: bool = False


class EmbeddingExtractor(ABC):
    """Template method for per-image embedding extraction."""

    def __init__(self, config: ExtractionConfig | None = None) -> None:
        self.config = config or ExtractionConfig()

    @abstractmethod
    def extract_one(self, image_path: Path) -> np.ndarray:
        """Extract a 1D embedding for one image."""

    def extract_split(
        self,
        metadata: pd.DataFrame,
        split: str,
        output_dir: Path | None = None,
    ) -> ExtractionResult:
        embeddings: list[np.ndarray] = []
        successful_rows: list[dict[str, object]] = []
        errors: list[dict[str, object]] = []
        start = time.time()
        total = len(metadata)

        for pos, row in metadata.iterrows():
            image_path = Path(str(row["image_path"]))
            if not image_path.exists():
                errors.append(self._error_row(split, row, pos, "FileNotFoundError: image not found"))
            else:
                try:
                    embedding = np.asarray(self.extract_one(image_path), dtype=np.float32)
                    if embedding.ndim != 1:
                        raise ValueError(f"Expected 1D embedding, got shape {embedding.shape}")
                    embeddings.append(embedding)
                    successful_rows.append(row.to_dict())
                except Exception as exc:
                    errors.append(self._error_row(split, row, pos, repr(exc)))

            processed = pos + 1
            if self.config.checkpoint_every and output_dir and processed % self.config.checkpoint_every == 0:
                self.save_checkpoint(output_dir, split, processed, embeddings, successful_rows, errors)

            if processed % 100 == 0 or processed == total:
                elapsed = time.time() - start
                print(
                    f"[{split}] {processed}/{total} processed | "
                    f"success: {len(embeddings)} | errors: {len(errors)} | {elapsed:.1f}s"
                )

        if not embeddings and not self.config.allow_empty:
            raise RuntimeError(f"No embeddings were extracted for split {split}")

        embeddings_array = (
            np.vstack(embeddings).astype(np.float32)
            if embeddings
            else np.empty((0, 0), dtype=np.float32)
        )
        success_metadata = pd.DataFrame(successful_rows).reset_index(drop=True)
        if not success_metadata.empty:
            success_metadata = success_metadata.drop(columns=["index"], errors="ignore")
            success_metadata.insert(0, "index", range(len(success_metadata)))
        return ExtractionResult(
            split=split,
            embeddings=embeddings_array,
            metadata=success_metadata,
            errors=pd.DataFrame(errors),
        )

    @staticmethod
    def _error_row(split: str, row: pd.Series, pos: int, error: str) -> dict[str, object]:
        image_path = Path(str(row.get("image_path", "")))
        return {
            "split": split,
            "metadata_index": row.get("index", pos),
            "image_path": str(image_path),
            "filename": row.get("filename", image_path.name),
            "error": error,
        }

    @staticmethod
    def save_checkpoint(
        output_dir: Path,
        split: str,
        processed_count: int,
        embeddings: list[np.ndarray],
        rows: list[dict[str, object]],
        errors: list[dict[str, object]],
    ) -> None:
        partial_dir = ensure_dir(Path(output_dir) / "partials")
        if embeddings:
            np.save(
                partial_dir / f"{split}_embeddings_until_{processed_count:06d}.npy",
                np.vstack(embeddings).astype(np.float32),
            )
            metadata = pd.DataFrame(rows).reset_index(drop=True)
            metadata = metadata.drop(columns=["index"], errors="ignore")
            metadata.insert(0, "index", range(len(metadata)))
            save_dataframe(metadata, partial_dir / f"{split}_metadata_until_{processed_count:06d}.csv")
        if errors:
            save_dataframe(pd.DataFrame(errors), partial_dir / f"{split}_errors_until_{processed_count:06d}.csv")
