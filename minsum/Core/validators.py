"""Validation helpers for metadata and embedding artifacts."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .schemas import REQUIRED_METADATA_COLUMNS


def ensure_columns(frame: pd.DataFrame, required: Iterable[str], frame_name: str) -> None:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def validate_metadata(
    metadata: pd.DataFrame,
    frame_name: str = "metadata",
    required_columns: Iterable[str] = REQUIRED_METADATA_COLUMNS,
) -> None:
    ensure_columns(metadata, required_columns, frame_name)
    if metadata["person_id"].isna().any():
        raise ValueError(f"{frame_name} contains missing person_id values")
    if metadata["filename"].duplicated().any():
        duplicated = metadata.loc[metadata["filename"].duplicated(), "filename"].head().tolist()
        raise ValueError(f"{frame_name} contains duplicated filename values, e.g. {duplicated}")


def validate_embeddings_alignment(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    split_name: str = "split",
) -> None:
    if embeddings.ndim != 2:
        raise ValueError(f"{split_name} embeddings must be 2D, got shape {embeddings.shape}")
    if embeddings.shape[0] != len(metadata):
        raise ValueError(
            f"{split_name} embeddings/metadata mismatch: "
            f"{embeddings.shape[0]} vectors vs {len(metadata)} metadata rows"
        )


def validate_finite_embeddings(embeddings: np.ndarray, split_name: str = "split") -> None:
    if np.isnan(embeddings).any():
        raise ValueError(f"{split_name} embeddings contain NaN values")
    if np.isinf(embeddings).any():
        raise ValueError(f"{split_name} embeddings contain Inf values")


def validate_embedding_bundle(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    split_name: str,
) -> None:
    validate_metadata(metadata, f"{split_name}_metadata")
    validate_embeddings_alignment(embeddings, metadata, split_name)
    validate_finite_embeddings(embeddings, split_name)
