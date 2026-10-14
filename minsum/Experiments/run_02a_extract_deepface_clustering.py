"""Extract DeepFace embeddings for clustering."""

from __future__ import annotations

import argparse
from pathlib import Path

from minsum.Core.paths import ProjectPaths
from minsum.Preprocessing.clustering_embeddings.deepface_extractor import DeepFaceConfig, DeepFaceEmbeddingExtractor
from minsum.Experiments.run_utils import (
    extraction_output_dir,
    load_metadata_splits,
    save_extraction_results,
    select_metadata_shards,
)


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--project-root", type=Path, required=True)
    cli.add_argument("--dataset-name", default="rfw")
    cli.add_argument("--embedding-name", default="deepface_arcface")
    cli.add_argument("--metadata-dir", type=Path, default=None)
    cli.add_argument("--output-dir", type=Path, default=None)
    cli.add_argument("--splits", default="train,val")
    cli.add_argument("--model-name", default="ArcFace")
    cli.add_argument("--detector-backend", default="skip")
    cli.add_argument("--align", action="store_true")
    cli.add_argument("--enforce-detection", action="store_true")
    cli.add_argument("--normalization", default="base")
    cli.add_argument("--checkpoint-every", type=int, default=0)
    cli.add_argument("--shard-index", type=int, default=None)
    cli.add_argument("--num-shards", type=int, default=1)
    args = cli.parse_args()

    split_names = tuple(split.strip() for split in args.splits.split(",") if split.strip())
    metadata = load_metadata_splits(args.project_root, args.dataset_name, split_names, args.metadata_dir)
    metadata = select_metadata_shards(metadata, args.shard_index, args.num_shards)
    base_output_dir = args.output_dir or ProjectPaths(args.project_root, args.dataset_name).embeddings_dir(
        "clustering",
        args.embedding_name,
    )
    output_dir = extraction_output_dir(base_output_dir, args.shard_index, args.num_shards)

    extractor = DeepFaceEmbeddingExtractor(
        DeepFaceConfig(
            model_name=args.model_name,
            detector_backend=args.detector_backend,
            align=args.align,
            enforce_detection=args.enforce_detection,
            normalization=args.normalization,
            checkpoint_every=args.checkpoint_every,
        )
    )
    results = [
        extractor.extract_split(frame, split=split, output_dir=output_dir)
        for split, frame in metadata.items()
    ]
    save_extraction_results(
        output_dir,
        results,
        {
            "dataset_name": args.dataset_name,
            "task": "clustering_embedding_extraction",
            "embedding_name": args.embedding_name,
            "library": "deepface",
            "model_name": args.model_name,
            "detector_backend": args.detector_backend,
            "align": args.align,
            "enforce_detection": args.enforce_detection,
            "normalization": args.normalization,
            "project_root": str(args.project_root),
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
        },
    )
    print(f"DeepFace clustering embeddings saved to: {output_dir}")


if __name__ == "__main__":
    main()
