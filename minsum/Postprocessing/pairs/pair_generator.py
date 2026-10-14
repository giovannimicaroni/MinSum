"""Generate same-person and different-person pairs from clustered metadata."""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import pandas as pd


@dataclass
class PairGenerationConfig:
    per_cluster: bool = True
    same_ethnicity_for_diff: bool = True
    max_pairs_per_cluster: int = 1000
    random_seed: int = 42
    person_col: str = "person_id"
    cluster_col: str = "cluster"
    image_col: str = "image_path"
    filename_col: str = "filename"
    diff_constraint_cols: tuple[str, ...] = ("ethnicity_id",)


class PairGenerator:
    """Generate pair CSVs using metadata columns instead of filename heuristics."""

    def __init__(self, config: PairGenerationConfig | None = None) -> None:
        self.config = config or PairGenerationConfig()
        self.random = random.Random(self.config.random_seed)

    def generate(self, clusters_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        self._validate_input(clusters_df)
        groups = self._groups(clusters_df)
        same_rows: list[dict[str, object]] = []
        diff_rows: list[dict[str, object]] = []
        for _, group in groups:
            group_same = self._same_person_pairs(group)
            if not group_same:
                continue

            group_diff = self._different_person_pairs(group, limit=len(group_same))
            balanced_count = min(len(group_same), len(group_diff))
            if balanced_count == 0:
                continue

            if len(group_same) > balanced_count:
                self.random.shuffle(group_same)
                group_same = group_same[:balanced_count]
            if len(group_diff) > balanced_count:
                self.random.shuffle(group_diff)
                group_diff = group_diff[:balanced_count]

            same_rows.extend(group_same)
            diff_rows.extend(group_diff)
        same = self._pair_frame(same_rows)
        diff = self._pair_frame(diff_rows)
        return same.reset_index(drop=True), diff.reset_index(drop=True)


    @staticmethod
    def _pair_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
        columns = [
            "image_a",
            "image_b",
            "filename_a",
            "filename_b",
            "person_id_a",
            "person_id_b",
            "cluster_a",
            "cluster_b",
            "same_person",
            "pair_label",
        ]
        if not rows:
            return pd.DataFrame(columns=columns)
        frame = pd.DataFrame(rows)
        for column in columns:
            if column not in frame.columns:
                frame[column] = None
        return frame.drop_duplicates(subset=["filename_a", "filename_b"])

    def _groups(self, clusters_df: pd.DataFrame) -> Iterable[tuple[object, pd.DataFrame]]:
        if self.config.per_cluster:
            return clusters_df.groupby(self.config.cluster_col, dropna=False)
        return [(None, clusters_df)]

    def _same_person_pairs(self, group: pd.DataFrame) -> list[dict[str, object]]:
        limit = self.config.max_pairs_per_cluster
        sampled_pairs: list[tuple[object, object]] = []
        pairs_seen = 0
        for _, person_group in group.groupby(self.config.person_col):
            if len(person_group) < 2:
                continue
            for pair in combinations(person_group.index.tolist(), 2):
                pairs_seen += 1
                if len(sampled_pairs) < limit:
                    sampled_pairs.append(pair)
                    continue
                replacement_index = self.random.randrange(pairs_seen)
                if replacement_index < limit:
                    sampled_pairs[replacement_index] = pair

        self.random.shuffle(sampled_pairs)
        return [
            self._row(group.loc[idx_a], group.loc[idx_b], pair_label=1)
            for idx_a, idx_b in sampled_pairs
        ]

    def _different_person_pairs(
        self,
        group: pd.DataFrame,
        limit: int,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        candidate_groups = [group]
        if self.config.same_ethnicity_for_diff:
            available = [col for col in self.config.diff_constraint_cols if col in group.columns]
            if available:
                candidate_groups = [sub for _, sub in group.groupby(available, dropna=False)]

        candidate_indices = [
            candidate_group.index.tolist()
            for candidate_group in candidate_groups
            if len(candidate_group) >= 2
        ]
        if not candidate_indices:
            return rows

        weights = [len(indices) * (len(indices) - 1) / 2 for indices in candidate_indices]
        seen: set[frozenset[str]] = set()
        max_attempts = max(limit * 100, 100)
        attempts = 0
        while len(rows) < limit and attempts < max_attempts:
            indices = self.random.choices(candidate_indices, weights=weights, k=1)[0]
            idx_a, idx_b = self.random.sample(indices, 2)
            row_a = group.loc[idx_a]
            row_b = group.loc[idx_b]
            attempts += 1
            if row_a[self.config.person_col] == row_b[self.config.person_col]:
                continue
            key = frozenset({str(row_a[self.config.filename_col]), str(row_b[self.config.filename_col])})
            if key in seen:
                continue
            seen.add(key)
            rows.append(self._row(row_a, row_b, pair_label=0))
        return rows

    def _row(self, row_a: pd.Series, row_b: pd.Series, pair_label: int) -> dict[str, object]:
        cfg = self.config
        output: dict[str, object] = {
            "image_a": row_a[cfg.image_col],
            "image_b": row_b[cfg.image_col],
            "filename_a": row_a[cfg.filename_col],
            "filename_b": row_b[cfg.filename_col],
            "person_id_a": row_a[cfg.person_col],
            "person_id_b": row_b[cfg.person_col],
            "cluster_a": row_a[cfg.cluster_col] if cfg.cluster_col in row_a else None,
            "cluster_b": row_b[cfg.cluster_col] if cfg.cluster_col in row_b else None,
            "same_person": bool(row_a[cfg.person_col] == row_b[cfg.person_col]),
            "pair_label": int(pair_label),
        }
        for col in cfg.diff_constraint_cols + ("gender", "age"):
            if col in row_a.index:
                output[f"{col}_a"] = row_a[col]
                output[f"{col}_b"] = row_b[col]
        return output

    def _validate_input(self, clusters_df: pd.DataFrame) -> None:
        required = [
            self.config.person_col,
            self.config.image_col,
            self.config.filename_col,
        ]
        if self.config.per_cluster:
            required.append(self.config.cluster_col)
        missing = [col for col in required if col not in clusters_df.columns]
        if missing:
            raise ValueError(f"Cluster dataframe is missing columns: {missing}")
