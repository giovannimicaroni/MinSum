"""Column and artifact schemas shared by the pipeline.

The pipeline is intentionally dataset-agnostic. Dataset-specific fields such as
ethnicity_id are optional attributes; person_id is the only identity contract
required for pair generation and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


REQUIRED_METADATA_COLUMNS: tuple[str, ...] = (
    "index",
    "image_path",
    "relative_path",
    "filename",
    "stem",
    "extension",
    "dataset",
    "split",
    "parser_name",
    "person_id",
)

OPTIONAL_ATTRIBUTE_COLUMNS: tuple[str, ...] = (
    "ethnicity_id",
    "identity_id",
    "image_number",
    "gender",
    "age",
    "source",
)

CLUSTER_COLUMNS: tuple[str, ...] = (
    "cluster",
    "similarity_to_centroid",
    "cosine_distance_to_centroid",
)


@dataclass(frozen=True)
class EmbeddingFileNames:
    """Standard filenames for split-aligned embedding artifacts."""

    embeddings_suffix: str = "_embeddings.npy"
    metadata_suffix: str = "_metadata.csv"
    errors: str = "extraction_errors.csv"
    config: str = "config.json"

    def embeddings(self, split: str) -> str:
        return f"{split}{self.embeddings_suffix}"

    def metadata(self, split: str) -> str:
        return f"{split}{self.metadata_suffix}"


@dataclass(frozen=True)
class PairColumns:
    """Standard pair columns used across distance and evaluation stages."""

    image_a: str = "image_a"
    image_b: str = "image_b"
    filename_a: str = "filename_a"
    filename_b: str = "filename_b"
    person_id_a: str = "person_id_a"
    person_id_b: str = "person_id_b"
    cluster_a: str = "cluster_a"
    cluster_b: str = "cluster_b"
    same_person: str = "same_person"
    pair_label: str = "pair_label"
    distance: str = "cosine_distance"


EMBEDDING_FILE_NAMES = EmbeddingFileNames()
PAIR_COLUMNS = PairColumns()


def metadata_attribute_columns(columns: Sequence[str]) -> list[str]:
    """Return optional dataset attributes present in a metadata dataframe."""

    return [col for col in OPTIONAL_ATTRIBUTE_COLUMNS if col in columns]
