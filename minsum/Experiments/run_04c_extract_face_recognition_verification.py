"""Extract dlib face recognition embeddings for verification distances."""

from __future__ import annotations

import argparse
from pathlib import Path

from minsum.Core.paths import ProjectPaths
from minsum.Experiments.run_utils import (
    extraction_output_dir,
    load_metadata_splits,
    save_extraction_results,
    select_metadata_shards,
)
from minsum.VerificationEmbeddings.face_recognition import (
    FaceRecognitionConfig,
    FaceRecognitionExtractor,
)


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value: true/false")


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--project-root", type=Path, required=True)
    cli.add_argument("--dataset-name", default="rfw")
    cli.add_argument("--embedding-name", default="face_recognition_dlib")
    cli.add_argument("--metadata-dir", type=Path, default=None)
    cli.add_argument("--output-dir", type=Path, default=None)
    cli.add_argument("--splits", default="train,val")
    cli.add_argument(
        "--mode",
        choices=("prealigned", "detect"),
        default="prealigned",
        help="prealigned treats the whole image as the face crop; detect runs dlib face detection first.",
    )
    cli.add_argument("--detector-model", choices=("hog", "cnn"), default="hog")
    cli.add_argument("--landmark-model", choices=("small", "large"), default="small")
    cli.add_argument("--num-jitters", type=int, default=1)
    cli.add_argument(
        "--normalize",
        type=parse_bool,
        default=True,
        help="Whether to L2-normalize dlib descriptors before saving. Use false for canonical dlib Euclidean distance.",
    )
    cli.add_argument("--checkpoint-every", type=int, default=100)
    cli.add_argument("--shard-index", type=int, default=None)
    cli.add_argument("--num-shards", type=int, default=1)
    args = cli.parse_args()

    split_names = tuple(split.strip() for split in args.splits.split(",") if split.strip())
    metadata = load_metadata_splits(args.project_root, args.dataset_name, split_names, args.metadata_dir)
    metadata = select_metadata_shards(metadata, args.shard_index, args.num_shards)
    base_output_dir = args.output_dir or ProjectPaths(args.project_root, args.dataset_name).embeddings_dir(
        "verification",
        args.embedding_name,
    )
    output_dir = extraction_output_dir(base_output_dir, args.shard_index, args.num_shards)

    extractor = FaceRecognitionExtractor(
        FaceRecognitionConfig(
            mode=args.mode,
            detector_model=args.detector_model,
            landmark_model=args.landmark_model,
            num_jitters=args.num_jitters,
            normalize=args.normalize,
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
            "task": "verification_embedding_extraction",
            "embedding_name": args.embedding_name,
            "library": "dlib",
            "backend": "dlib",
            "mode": args.mode,
            "detector_model": args.detector_model,
            "landmark_model": args.landmark_model,
            "num_jitters": args.num_jitters,
            "embedding_dim": 128,
            "normalized": args.normalize,
            "project_root": str(args.project_root),
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
        },
    )
    print(f"face_recognition verification embeddings saved to: {output_dir}")
    print(f"face_recognition extraction mode: {args.mode}")


if __name__ == "__main__":
    main()
