"""Build standardized metadata for a dataset.

Examples:
  python -m minsum.Experiments.run_01_build_metadata \
    --project-root /home/emily.tomadon/dataset \
    --dataset-root /home/emily.tomadon/dataset/datasets/rfw_split/RFWsummarized_split \
    --dataset-name rfw \
    --parser rfw
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from minsum.Core.artifact_store import ensure_dir, save_dataframe, save_json
from minsum.Core.paths import ProjectPaths
from minsum.Datasets.generic_folder import GenericFolderParser
from minsum.Datasets.identity_split import IdentitySplitConfig, split_metadata_by_identity
from minsum.Datasets.rfw import RFWParser


def build_parser(args: argparse.Namespace):
    if args.parser == "rfw":
        return RFWParser(dataset_name=args.dataset_name)
    if args.parser == "generic-folder":
        return GenericFolderParser(dataset_name=args.dataset_name)
    raise ValueError(f"Unknown parser: {args.parser}")


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--project-root", type=Path, required=True)
    cli.add_argument("--dataset-root", type=Path, required=True)
    cli.add_argument("--dataset-name", default="rfw")
    cli.add_argument("--parser", choices=["rfw", "generic-folder"], default="rfw")
    cli.add_argument("--splits", default="train,val", help="Comma-separated split names.")
    cli.add_argument(
        "--split-by-identity",
        action="store_true",
        help=(
            "Treat dataset-root as an unsplit image tree and create identity-disjoint "
            "train/val/test metadata without moving images."
        ),
    )
    cli.add_argument("--train-ratio", type=float, default=0.70)
    cli.add_argument("--val-ratio", type=float, default=0.15)
    cli.add_argument("--test-ratio", type=float, default=0.15)
    cli.add_argument("--random-seed", type=int, default=42)
    cli.add_argument("--output-dir", type=Path, default=None)
    args = cli.parse_args()

    parser = build_parser(args)
    output_dir = args.output_dir or ProjectPaths(args.project_root, args.dataset_name).metadata_dir()
    ensure_dir(output_dir)

    identity_manifest = None
    unassigned_errors = pd.DataFrame()
    if args.split_by_identity:
        raw_metadata, unassigned_errors = parser.build_split_metadata(
            split="unassigned",
            split_dir=args.dataset_root,
            project_root=args.project_root,
        )
        metadata_by_split, all_metadata, identity_manifest = split_metadata_by_identity(
            raw_metadata,
            IdentitySplitConfig(
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                random_seed=args.random_seed,
            ),
        )
        splits = {split: args.dataset_root for split in metadata_by_split}
        errors_by_split = {split: pd.DataFrame() for split in metadata_by_split}
    else:
        split_names = [split.strip() for split in args.splits.split(",") if split.strip()]
        splits = {split: args.dataset_root / split for split in split_names}
        metadata_by_split, errors_by_split, all_metadata = parser.build_metadata(
            splits=splits,
            project_root=args.project_root,
        )

    for split, metadata in metadata_by_split.items():
        save_dataframe(metadata, output_dir / f"{split}_metadata.csv")
        errors = errors_by_split[split]
        if not errors.empty:
            save_dataframe(errors, output_dir / f"{split}_metadata_errors.csv")

    save_dataframe(all_metadata, output_dir / "all_metadata.csv")
    if identity_manifest is not None:
        save_dataframe(identity_manifest, output_dir / "identity_split.csv")
    if not unassigned_errors.empty:
        save_dataframe(unassigned_errors, output_dir / "metadata_errors.csv")

    report_rows = []
    for split, metadata in metadata_by_split.items():
        row = {
            "dataset": args.dataset_name,
            "split": split,
            "split_dir": str(splits[split]),
            "num_valid_images": len(metadata),
            "num_errors": len(errors_by_split[split]),
            "num_identities": metadata["person_id"].nunique() if "person_id" in metadata else 0,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        for attr in ("ethnicity_id", "gender", "age"):
            if attr in metadata:
                row[f"num_{attr}"] = metadata[attr].nunique()
        report_rows.append(row)
    save_dataframe(pd.DataFrame(report_rows), output_dir / "metadata_build_report.csv")

    save_json(
        {
            "dataset_name": args.dataset_name,
            "parser_name": parser.parser_name,
            "project_root": str(args.project_root),
            "dataset_root": str(args.dataset_root),
            "splits": {split: str(path) for split, path in splits.items()},
            "output_dir": str(output_dir),
            "split_by_identity": args.split_by_identity,
            "identity_split": {
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
                "test_ratio": args.test_ratio,
                "random_seed": args.random_seed,
            }
            if args.split_by_identity
            else None,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        output_dir / "config.json",
    )

    print(f"Metadata saved to: {output_dir}")
    print(f"Total images: {len(all_metadata)}")
    if identity_manifest is not None:
        split_counts = identity_manifest["split"].value_counts()
        for split in ("train", "val", "test"):
            print(
                f"[{split}] identities: {int(split_counts.get(split, 0))} | "
                f"images: {len(metadata_by_split[split])}"
            )


if __name__ == "__main__":
    main()
