"""Compute pair distances using verification embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from minsum.Core.embedding_store import EmbeddingStore, build_embedding_index
from .cosine import cosine_distance, euclidean_distance


@dataclass
class PairDistanceCalculator:
    embedding_index: dict[str, np.ndarray]
    verification_embedding_name: str
    distance_metric: str = "cosine"
    key_col: str = "filename"

    @classmethod
    def from_embedding_dir(
        cls,
        embedding_dir: Path,
        split: str,
        verification_embedding_name: str,
        distance_metric: str = "cosine",
        key_col: str = "filename",
    ) -> "PairDistanceCalculator":
        split_data = EmbeddingStore(embedding_dir).load_split(split)
        index = build_embedding_index(split_data.embeddings, split_data.metadata, key_col=key_col)
        return cls(
            index,
            verification_embedding_name=verification_embedding_name,
            distance_metric=distance_metric,
            key_col=key_col,
        )

    @property
    def distance_col(self) -> str:
        if self.distance_metric == "cosine":
            return "cosine_distance"
        if self.distance_metric == "euclidean":
            return "euclidean_distance"
        raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

    def _distance(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        if self.distance_metric == "cosine":
            return cosine_distance(emb_a, emb_b)
        if self.distance_metric == "euclidean":
            return euclidean_distance(emb_a, emb_b)
        raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

    def compute(self, pairs: pd.DataFrame) -> pd.DataFrame:
        result = pairs.copy()
        distances = []
        for _, row in result.iterrows():
            key_a = str(row.get("filename_a", row.get("image_a")))
            key_b = str(row.get("filename_b", row.get("image_b")))
            emb_a = self.embedding_index.get(key_a)
            emb_b = self.embedding_index.get(key_b)
            if emb_a is None or emb_b is None:
                distances.append(float("nan"))
            else:
                distances.append(self._distance(emb_a, emb_b))
        result[self.distance_col] = distances
        result["distance_metric"] = self.distance_metric
        result["verification_embedding"] = self.verification_embedding_name
        return result
