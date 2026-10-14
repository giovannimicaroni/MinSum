"""Fallback parser for folder-per-identity datasets."""

from __future__ import annotations

from pathlib import Path

from .base import DatasetParser


class GenericFolderParser(DatasetParser):
    """Infer person_id from the image parent directory.

    This is useful for datasets laid out as split/person_id/image.jpg. It is a
    safe fallback for future datasets that do not encode identity in filenames.
    """

    def __init__(self, dataset_name: str, parser_name: str = "generic_folder") -> None:
        super().__init__(dataset_name=dataset_name, parser_name=parser_name)

    def parse_image(self, image_path: Path, split_dir: Path) -> dict[str, str]:
        try:
            relative = image_path.relative_to(split_dir)
            person_id = relative.parts[0] if len(relative.parts) > 1 else image_path.stem
            image_id = relative.as_posix()
        except ValueError:
            person_id = image_path.parent.name
            image_id = f"{person_id}/{image_path.name}"
        return {
            "person_id": person_id,
            "image_id": image_id,
            "filename": image_id,
        }
