"""Cluster report builders."""

from __future__ import annotations

import pandas as pd


def make_cluster_report(
    clusters_df: pd.DataFrame,
    split_name: str,
    cluster_col: str = "cluster",
    attribute_columns: tuple[str, ...] = ("ethnicity_id", "gender", "age"),
) -> pd.DataFrame:
    if cluster_col not in clusters_df.columns:
        raise ValueError(f"Missing cluster column {cluster_col!r}")

    aggregations: dict[str, tuple[str, str]] = {
        "num_images": ("filename", "count"),
    }
    if "person_id" in clusters_df.columns:
        aggregations["num_identities"] = ("person_id", "nunique")
    if "similarity_to_centroid" in clusters_df.columns:
        aggregations.update(
            {
                "mean_similarity_to_centroid": ("similarity_to_centroid", "mean"),
                "median_similarity_to_centroid": ("similarity_to_centroid", "median"),
                "min_similarity_to_centroid": ("similarity_to_centroid", "min"),
                "max_similarity_to_centroid": ("similarity_to_centroid", "max"),
            }
        )
    if "cosine_distance_to_centroid" in clusters_df.columns:
        aggregations.update(
            {
                "mean_cosine_distance_to_centroid": ("cosine_distance_to_centroid", "mean"),
                "median_cosine_distance_to_centroid": ("cosine_distance_to_centroid", "median"),
                "min_cosine_distance_to_centroid": ("cosine_distance_to_centroid", "min"),
                "max_cosine_distance_to_centroid": ("cosine_distance_to_centroid", "max"),
            }
        )

    report = clusters_df.groupby(cluster_col).agg(**aggregations).reset_index().sort_values(cluster_col)

    for attr in attribute_columns:
        if attr not in clusters_df.columns:
            continue
        counts = clusters_df.groupby([cluster_col, attr]).size().unstack(fill_value=0)
        counts.columns = [f"{attr}_{value}_count" for value in counts.columns]
        report = report.merge(counts.reset_index(), on=cluster_col, how="left")

    report.insert(0, "split", split_name)
    return report
