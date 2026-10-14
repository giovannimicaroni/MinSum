"""Analyze intra-cluster impostor similarity from saved pair distances."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from minsum.Core.artifact_store import ensure_dir, save_dataframe, save_json
from minsum.Core.paths import ProjectPaths
from minsum.Experiments.run_utils import parse_csv_list


def _similarity_from_distance(frame: pd.DataFrame, distance_col: str) -> pd.Series:
    distances = frame[distance_col].astype(float)
    if distance_col == "cosine_distance":
        return 1.0 - distances
    if distance_col == "euclidean_distance":
        return -distances
    raise ValueError(
        f"Unsupported distance column {distance_col!r}. "
        "Expected cosine_distance or euclidean_distance."
    )


def _cluster_similarity_stats(
    train_distances: pd.DataFrame,
    distance_col: str,
    min_impostor_pairs: int,
) -> pd.DataFrame:
    required = {"cluster_a", "cluster_b", "pair_label", distance_col}
    missing = required.difference(train_distances.columns)
    if missing:
        raise ValueError(f"Missing required columns in train distances: {sorted(missing)}")

    frame = train_distances.dropna(subset=[distance_col, "pair_label", "cluster_a", "cluster_b"]).copy()
    frame["cluster_a"] = frame["cluster_a"].astype(str)
    frame["cluster_b"] = frame["cluster_b"].astype(str)
    frame["pair_label"] = frame["pair_label"].astype(int)
    frame["similarity"] = _similarity_from_distance(frame, distance_col)

    same_cluster = frame[frame["cluster_a"] == frame["cluster_b"]].copy()
    impostors = same_cluster[same_cluster["pair_label"] == 0].copy()
    genuines = same_cluster[same_cluster["pair_label"] == 1].copy()

    impostor_stats = (
        impostors.groupby("cluster_a")
        .agg(
            intra_cluster_avg_impostor_similarity=("similarity", "mean"),
            intra_cluster_std_impostor_similarity=("similarity", "std"),
            intra_cluster_max_impostor_similarity=("similarity", "max"),
            intra_cluster_min_impostor_similarity=("similarity", "min"),
            num_impostor_pairs=("similarity", "size"),
            avg_impostor_distance=(distance_col, "mean"),
        )
        .reset_index()
        .rename(columns={"cluster_a": "cluster"})
    )
    genuine_stats = (
        genuines.groupby("cluster_a")
        .agg(
            intra_cluster_avg_genuine_similarity=("similarity", "mean"),
            intra_cluster_std_genuine_similarity=("similarity", "std"),
            num_genuine_pairs=("similarity", "size"),
            avg_genuine_distance=(distance_col, "mean"),
        )
        .reset_index()
        .rename(columns={"cluster_a": "cluster"})
    )

    stats = impostor_stats.merge(genuine_stats, on="cluster", how="outer")
    stats["cluster"] = stats["cluster"].astype(str)
    stats["impostor_minus_genuine_similarity"] = (
        stats["intra_cluster_avg_impostor_similarity"]
        - stats["intra_cluster_avg_genuine_similarity"]
    )
    stats["has_enough_impostor_pairs"] = stats["num_impostor_pairs"].fillna(0) >= min_impostor_pairs
    return stats.sort_values("cluster").reset_index(drop=True)


def _merge_thresholds(
    stats: pd.DataFrame,
    experiment_dir: Path,
    methods: list[str],
) -> pd.DataFrame:
    output = stats.copy()
    for method in methods:
        path = experiment_dir / method / "thresholds_by_cluster.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Thresholds not found for method {method!r}: {path}")
        thresholds = pd.read_csv(path)
        if not {"cluster", "threshold"}.issubset(thresholds.columns):
            raise ValueError(f"Threshold file has no cluster/threshold columns: {path}")
        thresholds = thresholds[["cluster", "threshold", "fpr", "fnr", "objective", "num_pairs"]].copy()
        thresholds["cluster"] = thresholds["cluster"].astype(str)
        thresholds = thresholds.rename(
            columns={
                "threshold": f"{method}_threshold",
                "fpr": f"{method}_train_fpr",
                "fnr": f"{method}_train_fnr",
                "objective": f"{method}_objective",
                "num_pairs": f"{method}_train_pairs",
            }
        )
        output = output.merge(thresholds, on="cluster", how="left")
    return output


def _summary(frame: pd.DataFrame, methods: list[str]) -> dict[str, object]:
    enough = frame[frame["has_enough_impostor_pairs"]].copy()
    target = enough if not enough.empty else frame
    summary: dict[str, object] = {
        "num_clusters": int(len(frame)),
        "num_clusters_with_enough_impostor_pairs": int(len(enough)),
        "total_impostor_pairs": int(frame["num_impostor_pairs"].fillna(0).sum()),
        "mean_intra_cluster_avg_impostor_similarity": float(
            target["intra_cluster_avg_impostor_similarity"].mean()
        ),
        "std_intra_cluster_avg_impostor_similarity": float(
            target["intra_cluster_avg_impostor_similarity"].std()
        ),
    }
    for method in methods:
        threshold_col = f"{method}_threshold"
        if threshold_col not in target.columns:
            continue
        valid = target.dropna(subset=["intra_cluster_avg_impostor_similarity", threshold_col])
        if len(valid) < 2:
            continue
        summary[f"pearson_iais_vs_{method}_threshold"] = float(
            valid["intra_cluster_avg_impostor_similarity"].corr(valid[threshold_col], method="pearson")
        )
        summary[f"spearman_iais_vs_{method}_threshold"] = float(
            valid["intra_cluster_avg_impostor_similarity"].corr(valid[threshold_col], method="spearman")
        )
    return summary


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--project-root", type=Path, required=True)
    cli.add_argument("--dataset-name", default="rfw")
    cli.add_argument("--cluster-embedding", required=True)
    cli.add_argument("--verification-embedding", required=True)
    cli.add_argument("--cluster-run-name", default="k30_cosine_centroid_no_pca")
    cli.add_argument("--distance-col", default="cosine_distance")
    cli.add_argument("--threshold-methods", default="minimum_sum")
    cli.add_argument("--analysis-name", default="impostor_similarity_analysis")
    cli.add_argument("--min-impostor-pairs", type=int, default=10)
    args = cli.parse_args()

    paths = ProjectPaths(args.project_root, args.dataset_name)
    experiment_dir = paths.experiment_dir(
        args.cluster_embedding,
        args.verification_embedding,
        args.cluster_run_name,
    )
    train_distances_path = experiment_dir / "train_pairs_with_distances.csv"
    if not train_distances_path.is_file():
        raise FileNotFoundError(f"Train pair distances not found: {train_distances_path}")

    methods = parse_csv_list(args.threshold_methods)
    if not methods:
        raise ValueError("--threshold-methods must contain at least one method")

    train_distances = pd.read_csv(train_distances_path)
    stats = _cluster_similarity_stats(
        train_distances,
        distance_col=args.distance_col,
        min_impostor_pairs=args.min_impostor_pairs,
    )
    merged = _merge_thresholds(stats, experiment_dir, methods)

    analysis_dir = ensure_dir(experiment_dir / args.analysis_name)
    save_dataframe(merged, analysis_dir / "cluster_impostor_similarity.csv")
    summary = _summary(merged, methods)
    save_json(summary, analysis_dir / "summary.json")
    save_dataframe(pd.DataFrame([summary]), analysis_dir / "summary.csv")

    top_difficult = merged.sort_values(
        "intra_cluster_avg_impostor_similarity",
        ascending=False,
    ).head(10)
    save_dataframe(top_difficult, analysis_dir / "top_impostor_similarity_clusters.csv")

    print(f"Analysis saved to: {analysis_dir}")
    print(pd.DataFrame([summary]).to_string(index=False))
    display_cols = [
        "cluster",
        "intra_cluster_avg_impostor_similarity",
        "num_impostor_pairs",
    ]
    for method in methods:
        threshold_col = f"{method}_threshold"
        if threshold_col in merged.columns:
            display_cols.append(threshold_col)
    print("\nTop clusters by intra-cluster impostor similarity:")
    print(top_difficult[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
