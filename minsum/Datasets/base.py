"""Generic dataset metadata building primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class DatasetParser:
    """Base class for converting image files into standardized metadata rows."""

    dataset_name: str
    parser_name: str
    image_extensions: set[str] = field(
        default_factory=lambda: {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    def list_images(self, split_dir: Path) -> list[Path]:
        split_dir = Path(split_dir)
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory does not exist: {split_dir}")
        return sorted(
            path
            for path in split_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in self.image_extensions
        )

    def parse_filename(self, filename: str) -> dict[str, str]:
        stem = Path(filename).stem
        return {"person_id": stem}

    def parse_image(self, image_path: Path, split_dir: Path) -> dict[str, str]:
        return self.parse_filename(image_path.name)

    def sort_columns(self, metadata: pd.DataFrame) -> list[str]:
        candidates = ["split", "person_id", "filename"]
        return [col for col in candidates if col in metadata.columns]

    def build_split_metadata(
        self,
        split: str,
        split_dir: Path,
        project_root: Path,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        rows: list[dict[str, object]] = []
        errors: list[dict[str, object]] = []

        for image_path in self.list_images(split_dir):
            try:
                parsed = self.parse_image(image_path, Path(split_dir))
                try:
                    relative_path = image_path.relative_to(project_root)
                except ValueError:
                    relative_path = image_path

                rows.append(
                    {
                        "image_path": str(image_path),
                        "relative_path": str(relative_path),
                        "filename": image_path.name,
                        "stem": image_path.stem,
                        "extension": image_path.suffix.lower(),
                        "dataset": self.dataset_name,
                        "split": split,
                        "parser_name": self.parser_name,
                        **parsed,
                    }
                )
            except Exception as exc:
                errors.append(
                    {
                        "image_path": str(image_path),
                        "filename": image_path.name,
                        "dataset": self.dataset_name,
                        "split": split,
                        "parser_name": self.parser_name,
                        "error": repr(exc),
                    }
                )

        metadata = pd.DataFrame(rows)
        if not metadata.empty:
            sort_cols = self.sort_columns(metadata)
            metadata = metadata.sort_values(sort_cols).reset_index(drop=True)
            metadata.insert(0, "index", range(len(metadata)))

        return metadata, pd.DataFrame(errors)

    def build_metadata(
        self,
        splits: dict[str, Path],
        project_root: Path,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
        metadata_by_split: dict[str, pd.DataFrame] = {}
        errors_by_split: dict[str, pd.DataFrame] = {}

        for split, split_dir in splits.items():
            metadata, errors = self.build_split_metadata(split, split_dir, project_root)
            metadata_by_split[split] = metadata
            errors_by_split[split] = errors

        all_metadata = pd.concat(metadata_by_split.values(), ignore_index=True)
        if not all_metadata.empty:
            all_metadata = all_metadata.drop(columns=["index"], errors="ignore")
            all_metadata.insert(0, "index", range(len(all_metadata)))

        return metadata_by_split, errors_by_split, all_metadata
