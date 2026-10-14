"""Canonical output paths for dataset-agnostic experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Build paths for the standard artifact layout.

    Expected layout:
      outputs/metadata/{dataset}
      outputs/embeddings/{dataset}/{purpose}/{embedding_name}
      outputs/clusters/{dataset}/{cluster_embedding}/{run_name}
      outputs/pairs/{dataset}/{cluster_embedding}_clusters/{run_name}
      outputs/experiments/{dataset}/{cluster_embedding}_cluster__{verification_embedding}_verify/{run_name}
    """

    project_root: Path
    dataset_name: str
    outputs_dir: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_root", Path(self.project_root))
        if self.outputs_dir is not None:
            object.__setattr__(self, "outputs_dir", Path(self.outputs_dir))

    @property
    def outputs_root(self) -> Path:
        return self.outputs_dir or self.project_root / "outputs"

    def metadata_dir(self) -> Path:
        return self.outputs_root / "metadata" / self.dataset_name

    def metadata_path(self, split: str) -> Path:
        return self.metadata_dir() / f"{split}_metadata.csv"

    def all_metadata_path(self) -> Path:
        return self.metadata_dir() / "all_metadata.csv"

    def embeddings_dir(self, purpose: str, embedding_name: str) -> Path:
        return self.outputs_root / "embeddings" / self.dataset_name / purpose / embedding_name

    def clusters_dir(self, cluster_embedding: str, run_name: str) -> Path:
        return self.outputs_root / "clusters" / self.dataset_name / cluster_embedding / run_name

    def pairs_dir(self, cluster_embedding: str, run_name: str) -> Path:
        return (
            self.outputs_root
            / "pairs"
            / self.dataset_name
            / f"{cluster_embedding}_clusters"
            / run_name
        )

    def experiment_dir(
        self,
        cluster_embedding: str,
        verification_embedding: str,
        run_name: str,
    ) -> Path:
        experiment_name = f"{cluster_embedding}_cluster__{verification_embedding}_verify"
        return self.outputs_root / "experiments" / self.dataset_name / experiment_name / run_name
