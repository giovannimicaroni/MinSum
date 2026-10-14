"""Merge deterministic embedding extraction shards into standard split artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from minsum.Core.artifact_store import (
    load_embedding_split,
    save_dataframe,
    save_embedding_split,
    save_json,
)
from minsum.Core.paths import ProjectPaths
from minsum.Core.validators import validate_embedding_bundle
from minsum.Experiments.run_utils import parse_csv_list


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--project-root", type=Path, required=True)
    cli.add_argument("--dataset-name", required=True)
    cli.add_argument("--purpose", choices=("clustering", "verification"), required=True)
    cli.add_argument("--embedding-name", required=True)
    cli.add_argument("--num-shards", type=int, required=True)
    cli.add_argument("--splits", default="train,val")
    cli.add_argument("--output-dir", type=Path, default=None)
    cli.add_argument("--allow-errors", action="store_true")
    args = cli.parse_args()

    if args.num_shards < 1:
        raise ValueError("num-shards must be at least 1")

    paths = ProjectPaths(args.project_root, args.dataset_name)
    output_dir = args.output_dir or paths.embeddings_dir(args.purpose, args.embedding_name)
    split_names = parse_csv_list(args.splits)
    all_errors = []
    shapes = {}

    for shard_index in range(args.num_shards):
        shard_dir = (
            output_dir
            / "shards"
            / f"shard_{shard_index:04d}_of_{args.num_shards:04d}"
        )
        errors_path = shard_dir / "extraction_errors.csv"
        if not errors_path.exists():
            raise FileNotFoundError(f"Missing shard errors file: {errors_path}")
        errors = pd.read_csv(errors_path)
        if not errors.empty:
            errors["shard_index"] = shard_index
            all_errors.append(errors)

    errors_df = pd.concat(all_errors, ignore_index=True) if all_errors else pd.DataFrame(
        columns=["split", "metadata_index", "image_path", "filename", "error", "shard_index"]
    )
    save_dataframe(errors_df, output_dir / "extraction_errors.csv")
    if not errors_df.empty and not args.allow_errors:
        raise RuntimeError(
            f"Cannot merge with {len(errors_df)} extraction errors. "
            "Inspect extraction_errors.csv or pass --allow-errors."
        )

    for split in split_names:
        embeddings_parts = []
        metadata_parts = []
        for shard_index in range(args.num_shards):
            shard_dir = (
                output_dir
                / "shards"
                / f"shard_{shard_index:04d}_of_{args.num_shards:04d}"
            )
            embeddings, metadata = load_embedding_split(shard_dir, split)
            embeddings_parts.append(embeddings)
            metadata_parts.append(metadata)

        embeddings = np.vstack(embeddings_parts).astype(np.float32)
        metadata = pd.concat(metadata_parts, ignore_index=True)
        metadata = metadata.drop(columns=["index"], errors="ignore")
        metadata.insert(0, "index", range(len(metadata)))
        validate_embedding_bundle(embeddings, metadata, split)

        expected_metadata = pd.read_csv(paths.metadata_path(split), dtype=str)
        split_errors = errors_df[errors_df["split"] == split] if not errors_df.empty else errors_df
        if len(metadata) + len(split_errors) != len(expected_metadata):
            raise RuntimeError(
                f"{split} merge coverage mismatch: {len(metadata)} embeddings + "
                f"{len(split_errors)} errors != {len(expected_metadata)} metadata rows"
            )

        save_embedding_split(output_dir, split, embeddings, metadata)
        shapes[split] = list(embeddings.shape)
        print(f"[{split}] merged shape: {embeddings.shape}")

    save_json(
        {
            "dataset_name": args.dataset_name,
            "purpose": args.purpose,
            "embedding_name": args.embedding_name,
            "num_shards": args.num_shards,
            "splits": list(split_names),
            "num_errors": int(len(errors_df)),
            "shapes": shapes,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        output_dir / "merged_config.json",
    )
    print(f"Merged embeddings saved to: {output_dir}")


if __name__ == "__main__":
    main()
