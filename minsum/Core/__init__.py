"""Shared contracts and filesystem helpers for the research pipeline."""

from .paths import ProjectPaths
from .schemas import (
    EMBEDDING_FILE_NAMES,
    OPTIONAL_ATTRIBUTE_COLUMNS,
    PAIR_COLUMNS,
    REQUIRED_METADATA_COLUMNS,
)

__all__ = [
    "ProjectPaths",
    "EMBEDDING_FILE_NAMES",
    "OPTIONAL_ATTRIBUTE_COLUMNS",
    "PAIR_COLUMNS",
    "REQUIRED_METADATA_COLUMNS",
]
