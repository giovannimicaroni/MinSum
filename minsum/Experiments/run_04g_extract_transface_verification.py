"""Extract TransFace verification embeddings from aligned face crops."""

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
from minsum.VerificationEmbeddings.transface import TransFaceConfig, TransFaceExtractor


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--project-root", type=Path, required=True)
    cli.add_argument("--dataset-name", default="rfw")
    cli.add_argument("--embedding-name", default="transface_vit_s")
    cli.add_argument("--weights-path", type=Path, required=True)
    cli.add_argument(
        "--network",
        choices=(
            "vit_s_dp005_mask_0",
            "vit_b_dp005_mask_005",
            "vit_l_dp005_mask_005",
            "vit_s",
            "vit_b",
        ),
        default="vit_s_dp005_mask_0",
    )
    cli.add_argument("--device", default=None, help="Torch device; auto-detected when omitted.")
    cli.add_argument(
        "--no-l2-normalize",
        action="store_true",
        help="Store raw features. By default TransFace features are L2-normalized.",
    )
    cli.add_argument("--metadata-dir", type=Path, default=None)
    cli.add_argument("--output-dir", type=Path, default=None)
    cli.add_argument("--splits", default="train,val")
    cli.add_argument("--checkpoint-every", type=int, default=100)
    cli.add_argument("--shard-index", type=int, default=None)
    cli.add_argument("--num-shards", type=int, default=1)
    args = cli.parse_args()

    split_names = tuple(split.strip() for split in args.splits.split(",") if split.strip())
    metadata = load_metadata_splits(
        args.project_root,
        args.dataset_name,
        split_names,
        args.metadata_dir,
    )
    metadata = select_metadata_shards(metadata, args.shard_index, args.num_shards)
    base_output_dir = args.output_dir or ProjectPaths(
        args.project_root,
        args.dataset_name,
    ).embeddings_dir("verification", args.embedding_name)
    output_dir = extraction_output_dir(base_output_dir, args.shard_index, args.num_shards)

    extractor = TransFaceExtractor(
        TransFaceConfig(
            weights_path=args.weights_path,
            network=args.network,
            device=args.device,
            checkpoint_every=args.checkpoint_every,
            l2_normalize=not args.no_l2_normalize,
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
            "library": "transface_official",
            "network": args.network,
            "weights_path": str(args.weights_path),
            "device": str(extractor.device),
            "image_size": [112, 112],
            "color_order": "RGB",
            "input_scale": "[-1, 1]",
            "l2_normalize": not args.no_l2_normalize,
            "input_alignment": "prealigned",
            "project_root": str(args.project_root),
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
        },
    )
    print(f"TransFace verification embeddings saved to: {output_dir}")
    print(f"TransFace device: {extractor.device}")


if __name__ == "__main__":
    main()
