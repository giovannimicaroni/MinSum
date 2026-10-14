"""RFW summarized split parser."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from minsum.Core.artifact_store import ensure_dir, save_dataframe, save_json
from .base import DatasetParser


class RFWParser(DatasetParser):
    """Parse RFW filenames such as 0_00000_01.jpg."""

    def __init__(self, dataset_name: str = "rfw") -> None:
        super().__init__(dataset_name=dataset_name, parser_name="rfw")

    def parse_filename(self, filename: str) -> dict[str, str]:
        stem = Path(filename).stem
        parts = stem.split("_")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid RFW filename {filename!r}; expected ethnicity_identity_imageNumber"
            )
        ethnicity_id, identity_id, image_number = parts
        return {
            "ethnicity_id": ethnicity_id,
            "identity_id": identity_id,
            "person_id": f"{ethnicity_id}_{identity_id}",
            "image_number": image_number,
        }

    def sort_columns(self, metadata: pd.DataFrame) -> list[str]:
        candidates = ["split", "ethnicity_id", "identity_id", "image_number", "filename"]
        return [col for col in candidates if col in metadata.columns]


def build_rfw_metadata(
    project_root: Path,
    dataset_root: Path,
    dataset_name: str = "rfw",
    output_dir: Path | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    """Build and optionally save train/val/all metadata for the RFW split."""

    project_root = Path(project_root)
    dataset_root = Path(dataset_root)
    output_dir = output_dir or project_root / "outputs" / "metadata" / dataset_name
    parser = RFWParser(dataset_name=dataset_name)
    splits = {"train": dataset_root / "train", "val": dataset_root / "val"}
    metadata_by_split, errors_by_split, all_metadata = parser.build_metadata(
        splits=splits,
        project_root=project_root,
    )

    ensure_dir(output_dir)
    for split, metadata in metadata_by_split.items():
        save_dataframe(metadata, output_dir / f"{split}_metadata.csv")
        errors = errors_by_split[split]
        if not errors.empty:
            save_dataframe(errors, output_dir / f"{split}_metadata_errors.csv")

    save_dataframe(all_metadata, output_dir / "all_metadata.csv")
    report = []
    for split, metadata in metadata_by_split.items():
        report.append(
            {
                "dataset": dataset_name,
                "split": split,
                "split_dir": str(splits[split]),
                "num_valid_images": len(metadata),
                "num_errors": len(errors_by_split[split]),
                "num_identities": metadata["person_id"].nunique() if "person_id" in metadata else 0,
                "num_ethnicities": metadata["ethnicity_id"].nunique()
                if "ethnicity_id" in metadata
                else None,
            }
        )
    save_dataframe(pd.DataFrame(report), output_dir / "metadata_build_report.csv")
    save_json(
        {
            "dataset_name": dataset_name,
            "parser_name": parser.parser_name,
            "project_root": str(project_root),
            "dataset_root": str(dataset_root),
            "splits": {split: str(path) for split, path in splits.items()},
            "output_dir": str(output_dir),
        },
        output_dir / "config.json",
    )
    return metadata_by_split, errors_by_split, all_metadata
