"""Shared helpers for experiment run scripts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from minsum.Core.artifact_store import (
    ensure_dir,
    save_dataframe,
    save_embedding_split,
    save_json,
)
from minsum.Core.paths import ProjectPaths
from minsum.Core.embedding_extraction import ExtractionResult


def parse_csv_list(value: str | None) -> tuple[str, ...]:
    if value is None or value == "":
        return tuple()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def load_metadata_splits(
    project_root: Path,
    dataset_name: str,
    splits: Iterable[str],
    metadata_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    paths = ProjectPaths(project_root, dataset_name)
    root = Path(metadata_dir) if metadata_dir is not None else paths.metadata_dir()
    output = {}
    for split in splits:
        path = root / f"{split}_metadata.csv"
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found for split {split!r}: {path}")
        output[split] = pd.read_csv(path, dtype=str)
    return output


def select_metadata_shards(
    metadata: dict[str, pd.DataFrame],
    shard_index: int | None,
    num_shards: int,
) -> dict[str, pd.DataFrame]:
    """Select deterministic contiguous metadata slices for an extraction job."""

    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    if shard_index is None:
        if num_shards != 1:
            raise ValueError("--shard-index is required when --num-shards is greater than 1")
        return metadata
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index must be between 0 and {num_shards - 1}")

    selected = {}
    for split, frame in metadata.items():
        quotient, remainder = divmod(len(frame), num_shards)
        start = shard_index * quotient + min(shard_index, remainder)
        size = quotient + (1 if shard_index < remainder else 0)
        selected[split] = frame.iloc[start : start + size].reset_index(drop=True)
        print(
            f"[{split}] shard {shard_index}/{num_shards}: "
            f"rows {start}:{start + size} ({size} images)"
        )
    return selected


def extraction_output_dir(
    base_output_dir: Path,
    shard_index: int | None,
    num_shards: int,
) -> Path:
    if shard_index is None:
        return Path(base_output_dir)
    return (
        Path(base_output_dir)
        / "shards"
        / f"shard_{shard_index:04d}_of_{num_shards:04d}"
    )


def save_extraction_results(
    output_dir: Path,
    results: list[ExtractionResult],
    config: dict[str, object],
) -> None:
    output_dir = ensure_dir(output_dir)
    generated_files: dict[str, str] = {}
    shapes: dict[str, list[int]] = {}
    errors = []

    for result in results:
        embeddings_path, metadata_path = save_embedding_split(
            output_dir,
            result.split,
            result.embeddings,
            result.metadata,
        )
        generated_files[f"{result.split}_embeddings"] = str(embeddings_path)
        generated_files[f"{result.split}_metadata"] = str(metadata_path)
        shapes[f"{result.split}_embeddings"] = list(result.embeddings.shape)
        if not result.errors.empty:
            errors.append(result.errors)

    errors_df = pd.concat(errors, ignore_index=True) if errors else pd.DataFrame(
        columns=["split", "metadata_index", "image_path", "filename", "error"]
    )
    errors_path = save_dataframe(errors_df, output_dir / "extraction_errors.csv")
    generated_files["extraction_errors"] = str(errors_path)

    config_payload = {
        **config,
        "output_dir": str(output_dir),
        "generated_files": generated_files,
        "shapes": shapes,
        "num_errors": int(len(errors_df)),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(config_payload, output_dir / "config.json")


def default_cluster_run_name(n_clusters: int, use_pca: bool, pca_components: int | None) -> str:
    if use_pca:
        return f"k{n_clusters}_cosine_centroid_pca{pca_components}"
    return f"k{n_clusters}_cosine_centroid_no_pca"
