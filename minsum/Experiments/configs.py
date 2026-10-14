"""Experiment-level configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentConfig:
    project_root: Path
    dataset_name: str
    cluster_embedding: str
    verification_embedding: str
    cluster_run_name: str = "k15_cosine_centroid_no_pca"
    random_seed: int = 42
