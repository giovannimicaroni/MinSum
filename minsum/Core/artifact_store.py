"""Small IO helpers for standard pipeline artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schemas import EMBEDDING_FILE_NAMES


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    return path


def load_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_dataframe(frame: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    frame.to_csv(path, index=False)
    return path


def load_dataframe(path: Path, dtype: str | type | None = str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=dtype)


def save_embedding_split(
    output_dir: Path,
    split: str,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
) -> tuple[Path, Path]:
    output_dir = ensure_dir(output_dir)
    embeddings_path = output_dir / EMBEDDING_FILE_NAMES.embeddings(split)
    metadata_path = output_dir / EMBEDDING_FILE_NAMES.metadata(split)
    np.save(embeddings_path, embeddings.astype(np.float32))
    metadata.to_csv(metadata_path, index=False)
    return embeddings_path, metadata_path


def load_embedding_split(output_dir: Path, split: str) -> tuple[np.ndarray, pd.DataFrame]:
    output_dir = Path(output_dir)
    embeddings = np.load(output_dir / EMBEDDING_FILE_NAMES.embeddings(split))
    metadata = pd.read_csv(output_dir / EMBEDDING_FILE_NAMES.metadata(split), dtype=str)
    return embeddings, metadata
