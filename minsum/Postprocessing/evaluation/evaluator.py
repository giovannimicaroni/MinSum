"""Apply thresholds and save metrics."""

from __future__ import annotations

import pandas as pd

from .metrics import classification_metrics


def apply_thresholds(
    pairs_with_distances: pd.DataFrame,
    thresholds_by_cluster: pd.DataFrame,
    group_col: str = "cluster_a",
    distance_col: str = "cosine_distance",
) -> pd.DataFrame:
    thresholds = {
        row["cluster"]: float(row["threshold"])
        for _, row in thresholds_by_cluster.iterrows()
    }
    result = pairs_with_distances.copy()
    result["threshold"] = result[group_col].map(thresholds)
    if result["threshold"].isna().any():
        missing = sorted(result.loc[result["threshold"].isna(), group_col].dropna().unique().tolist())
        raise ValueError(f"Missing thresholds for groups: {missing}")
    result["pred_same"] = (result[distance_col].astype(float) <= result["threshold"]).astype(int)
    return result


def evaluate_pairs(
    classified_pairs: pd.DataFrame,
    label_col: str = "pair_label",
    prediction_col: str = "pred_same",
    group_cols: tuple[str, ...] = ("cluster_a", "ethnicity_id_a", "gender_a"),
) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    outputs["global"] = pd.DataFrame(
        [classification_metrics(classified_pairs[label_col], classified_pairs[prediction_col])]
    )
    for group_col in group_cols:
        if group_col not in classified_pairs.columns:
            continue
        rows = []
        for value, group in classified_pairs.groupby(group_col, dropna=False):
            metrics = classification_metrics(group[label_col], group[prediction_col])
            rows.append({group_col: value, **metrics})
        outputs[f"by_{group_col}"] = pd.DataFrame(rows)
    return outputs
