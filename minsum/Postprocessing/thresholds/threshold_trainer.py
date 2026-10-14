"""Train fixed and minimum-sum thresholds per cluster or another grouping column."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .threshold_methods import select_fixed_threshold, select_minimum_sum


@dataclass
class ThresholdTrainingConfig:
    num_thresholds: int = 200
    fixed_threshold: float = 0.45
    fpr_weight: float = 1.0
    tie_break: str = "median"
    threshold_max: float | None = None
    group_col: str = "cluster_a"
    distance_col: str = "cosine_distance"
    label_col: str = "pair_label"
    same_cluster_only: bool = True


def _training_frame(
    pairs_with_distances: pd.DataFrame,
    config: ThresholdTrainingConfig,
) -> pd.DataFrame:
    df = pairs_with_distances.dropna(subset=[config.distance_col, config.label_col]).copy()
    if config.same_cluster_only and {"cluster_a", "cluster_b"}.issubset(df.columns):
        df = df[df["cluster_a"] == df["cluster_b"]]
    if df.empty:
        raise ValueError("No pairs available to train/evaluate thresholds")
    return df


def _threshold_rows(
    pairs_with_distances: pd.DataFrame,
    config: ThresholdTrainingConfig,
    method: str,
) -> pd.DataFrame:
    df = _training_frame(pairs_with_distances, config)
    if config.threshold_max is None:
        observed_max = float(df[config.distance_col].astype(float).max())
        threshold_max = max(1.0, observed_max)
    else:
        threshold_max = float(config.threshold_max)
    thresholds = np.linspace(0.0, threshold_max, config.num_thresholds)
    rows = []

    for group_value, group in df.groupby(config.group_col, dropna=False):
        distances = group[config.distance_col].astype(float).to_numpy()
        labels = group[config.label_col].astype(int).to_numpy()

        if method == "minimum_sum":
            selected = select_minimum_sum(
                distances,
                labels,
                thresholds,
                fpr_weight=config.fpr_weight,
                tie_break=config.tie_break,
            )
        elif method == "fixed":
            selected = select_fixed_threshold(distances, labels, config.fixed_threshold)
        else:
            raise ValueError(f"Unknown threshold method: {method}")

        rows.append(
            {
                "cluster": group_value,
                "threshold": selected["threshold"],
                "method": method,
                "fpr": selected["fpr"],
                "fnr": selected["fnr"],
                "fp": selected["fp"],
                "fn": selected["fn"],
                "tp": selected["tp"],
                "tn": selected["tn"],
                "objective": selected["objective"],
                "fpr_weight": selected.get("fpr_weight"),
                "tie_break": selected.get("tie_break"),
                "num_pairs": len(group),
                "num_tied_thresholds": selected.get("num_tied_thresholds"),
                "tied_threshold_min": selected.get("tied_threshold_min"),
                "tied_threshold_max": selected.get("tied_threshold_max"),
            }
        )

    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


def train_minimum_sum_thresholds_by_group(
    pairs_with_distances: pd.DataFrame,
    config: ThresholdTrainingConfig | None = None,
) -> pd.DataFrame:
    return _threshold_rows(
        pairs_with_distances,
        config or ThresholdTrainingConfig(),
        method="minimum_sum",
    )


def build_fixed_thresholds_by_group(
    pairs_with_distances: pd.DataFrame,
    config: ThresholdTrainingConfig | None = None,
) -> pd.DataFrame:
    return _threshold_rows(
        pairs_with_distances,
        config or ThresholdTrainingConfig(),
        method="fixed",
    )
