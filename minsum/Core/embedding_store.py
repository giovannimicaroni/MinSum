"""Load and index split-aligned embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from minsum.Core.artifact_store import load_embedding_split, save_embedding_split
from minsum.Core.validators import validate_embedding_bundle


@dataclass
class EmbeddingSplit:
    split: str
    embeddings: np.ndarray
    metadata: pd.DataFrame

    def validate(self) -> None:
        validate_embedding_bundle(self.embeddings, self.metadata, self.split)


class EmbeddingStore:
    """Read/write the standard train_embeddings.npy + train_metadata.csv format."""

    def __init__(self, embedding_dir: Path) -> None:
        self.embedding_dir = Path(embedding_dir)

    def load_split(self, split: str, validate: bool = True) -> EmbeddingSplit:
        embeddings, metadata = load_embedding_split(self.embedding_dir, split)
        split_data = EmbeddingSplit(split=split, embeddings=embeddings, metadata=metadata)
        if validate:
            split_data.validate()
        return split_data

    def save_split(self, split_data: EmbeddingSplit) -> tuple[Path, Path]:
        split_data.validate()
        return save_embedding_split(
            self.embedding_dir,
            split_data.split,
            split_data.embeddings,
            split_data.metadata,
        )

    def load_splits(self, splits: tuple[str, ...] = ("train", "val")) -> dict[str, EmbeddingSplit]:
        return {split: self.load_split(split) for split in splits}


def build_embedding_index(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    key_col: str = "filename",
) -> dict[str, np.ndarray]:
    """Create key -> embedding mapping from aligned arrays and metadata."""

    if key_col not in metadata.columns:
        raise ValueError(f"Metadata does not contain key column {key_col!r}")
    if metadata[key_col].duplicated().any():
        duplicates = metadata.loc[metadata[key_col].duplicated(), key_col].head().tolist()
        raise ValueError(f"Cannot build embedding index with duplicated keys: {duplicates}")
    if len(metadata) != embeddings.shape[0]:
        raise ValueError("Embeddings and metadata are not aligned")
    return {
        str(key): embeddings[pos]
        for pos, key in enumerate(metadata[key_col].astype(str).tolist())
    }
