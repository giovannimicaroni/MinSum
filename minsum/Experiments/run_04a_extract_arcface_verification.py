"""Extract InsightFace ArcFace embeddings for verification distances."""

from __future__ import annotations

import argparse
from pathlib import Path

from minsum.Core.paths import ProjectPaths
from minsum.VerificationEmbeddings.arcface.insightface_extractor import (
    InsightFaceArcFaceExtractor,
    InsightFaceConfig,
)
from minsum.Experiments.run_utils import (
    extraction_output_dir,
    load_metadata_splits,
    save_extraction_results,
    select_metadata_shards,
)


def parse_det_size(value: str) -> tuple[int, int]:
    parts = [part.strip() for part in value.lower().replace("x", ",").split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("det-size must look like 640,640 or 640x640")
    return int(parts[0]), int(parts[1])


def parse_image_size(value: str) -> tuple[int, int]:
    parts = [part.strip() for part in value.lower().replace("x", ",").split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("image-size must look like 112,112 or 112x112")
    return int(parts[0]), int(parts[1])


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--project-root", type=Path, required=True)
    cli.add_argument("--dataset-name", default="rfw")
    cli.add_argument("--embedding-name", default="arcface")
    cli.add_argument("--metadata-dir", type=Path, default=None)
    cli.add_argument("--output-dir", type=Path, default=None)
    cli.add_argument("--splits", default="train,val")
    cli.add_argument("--model-name", default="buffalo_l")
    cli.add_argument("--ctx-id", type=int, default=0)
    cli.add_argument("--det-size", type=parse_det_size, default=(640, 640))
    cli.add_argument(
        "--mode",
        choices=("detect", "prealigned"),
        default="detect",
        help=(
            "detect runs the full InsightFace detector before recognition; "
            "prealigned skips detection and feeds resized crops directly to the recognition model."
        ),
    )
    cli.add_argument(
        "--prealigned",
        action="store_true",
        help="Shortcut for --mode prealigned.",
    )
    cli.add_argument(
        "--model-root",
        type=Path,
        default=None,
        help="Directory containing the InsightFace model pack, e.g. ~/.insightface/models/buffalo_l.",
    )
    cli.add_argument("--recognition-model", default="w600k_r50.onnx")
    cli.add_argument("--image-size", type=parse_image_size, default=(112, 112))
    cli.add_argument("--checkpoint-every", type=int, default=100)
    cli.add_argument("--shard-index", type=int, default=None)
    cli.add_argument("--num-shards", type=int, default=1)
    args = cli.parse_args()

    mode = "prealigned" if args.prealigned else args.mode
    split_names = tuple(split.strip() for split in args.splits.split(",") if split.strip())
    metadata = load_metadata_splits(args.project_root, args.dataset_name, split_names, args.metadata_dir)
    metadata = select_metadata_shards(metadata, args.shard_index, args.num_shards)
    base_output_dir = args.output_dir or ProjectPaths(args.project_root, args.dataset_name).embeddings_dir(
        "verification",
        args.embedding_name,
    )
    output_dir = extraction_output_dir(base_output_dir, args.shard_index, args.num_shards)

    extractor = InsightFaceArcFaceExtractor(
        InsightFaceConfig(
            model_name=args.model_name,
            ctx_id=args.ctx_id,
            det_size=args.det_size,
            mode=mode,
            model_root=args.model_root,
            recognition_model=args.recognition_model,
            image_size=args.image_size,
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
            "library": "insightface",
            "model_name": args.model_name,
            "ctx_id": args.ctx_id,
            "det_size": list(args.det_size),
            "mode": mode,
            "model_root": str(args.model_root) if args.model_root is not None else None,
            "recognition_model": args.recognition_model,
            "image_size": list(args.image_size),
            "project_root": str(args.project_root),
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
        },
    )
    print(f"ArcFace verification embeddings saved to: {output_dir}")
    print(f"InsightFace extraction mode: {mode}")


if __name__ == "__main__":
    main()
