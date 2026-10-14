"""Thin orchestration helpers that compose the modular pipeline."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from minsum.Core.artifact_store import ensure_dir, save_dataframe
from minsum.Core.paths import ProjectPaths
from minsum.Postprocessing.distances.pair_distance_calculator import PairDistanceCalculator
from minsum.Postprocessing.evaluation.evaluator import apply_thresholds, evaluate_pairs
from minsum.Postprocessing.pairs.pair_generator import PairGenerationConfig, PairGenerator
from minsum.Postprocessing.thresholds.threshold_trainer import (
    ThresholdTrainingConfig,
    build_fixed_thresholds_by_group,
    train_minimum_sum_thresholds_by_group,
)
from .configs import ExperimentConfig


def _shared_validation_pairs(
    shared_pairs_path: Path,
    val_clusters: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a fixed validation pair set and attach this run's cluster metadata."""

    shared = pd.read_csv(shared_pairs_path, dtype=str)
    if "pair_label" not in shared.columns:
        raise ValueError(f"Shared validation pairs file has no pair_label column: {shared_pairs_path}")

    shared = shared[shared["pair_label"].isin(["0", "1", 0, 1])].copy()
    required_pair_cols = {"filename_a", "filename_b"}
    missing_pair_cols = required_pair_cols.difference(shared.columns)
    if missing_pair_cols:
        raise ValueError(
            f"Shared validation pairs file is missing columns: {sorted(missing_pair_cols)}"
        )

    if "filename" not in val_clusters.columns:
        raise ValueError("Validation clusters must contain a filename column")
    if val_clusters["filename"].duplicated().any():
        duplicated = val_clusters.loc[val_clusters["filename"].duplicated(), "filename"].head(5)
        raise ValueError(
            "Validation cluster filenames must be unique to reuse shared pairs. "
            f"Examples: {duplicated.tolist()}"
        )

    metadata_by_filename = val_clusters.set_index("filename", drop=False).to_dict(orient="index")
    optional_columns = [
        "image_path",
        "person_id",
        "cluster",
        "ethnicity_id",
        "gender",
        "age",
    ]

    rows: list[dict[str, object]] = []
    for _, pair in shared.iterrows():
        filename_a = str(pair["filename_a"])
        filename_b = str(pair["filename_b"])
        if filename_a not in metadata_by_filename or filename_b not in metadata_by_filename:
            raise ValueError(
                "Shared validation pairs reference filenames absent from this validation split: "
                f"{filename_a!r}, {filename_b!r}"
            )

        meta_a = metadata_by_filename[filename_a]
        meta_b = metadata_by_filename[filename_b]
        row = pair.to_dict()
        for column in optional_columns:
            if column not in meta_a or column not in meta_b:
                continue
            if column == "image_path":
                row["image_a"] = meta_a[column]
                row["image_b"] = meta_b[column]
            elif column == "person_id":
                row["person_id_a"] = meta_a[column]
                row["person_id_b"] = meta_b[column]
                row["same_person"] = meta_a[column] == meta_b[column]
            elif column == "cluster":
                row["cluster_a"] = meta_a[column]
                row["cluster_b"] = meta_b[column]
            else:
                row[f"{column}_a"] = meta_a[column]
                row[f"{column}_b"] = meta_b[column]
        row["pair_label"] = int(row["pair_label"])
        rows.append(row)

    pairs = pd.DataFrame(rows)
    same = pairs[pairs["pair_label"] == 1].reset_index(drop=True)
    diff = pairs[pairs["pair_label"] == 0].reset_index(drop=True)
    return same, diff


def _save_method_outputs(
    experiment_dir: Path,
    method_name: str,
    thresholds: pd.DataFrame,
    train_distances: pd.DataFrame,
    val_distances: pd.DataFrame,
    distance_col: str,
) -> None:
    method_dir = ensure_dir(experiment_dir / method_name)
    save_dataframe(thresholds, method_dir / "thresholds_by_cluster.csv")

    train_classified = apply_thresholds(train_distances, thresholds, distance_col=distance_col)
    val_classified = apply_thresholds(val_distances, thresholds, distance_col=distance_col)
    save_dataframe(train_classified, method_dir / "train_pairs_classified.csv")
    save_dataframe(val_classified, method_dir / "val_pairs_classified.csv")

    for split, classified in (("train", train_classified), ("val", val_classified)):
        metrics = evaluate_pairs(classified)
        save_dataframe(metrics["global"], method_dir / f"{split}_metrics_global.csv")
        if "by_cluster_a" in metrics:
            save_dataframe(metrics["by_cluster_a"], method_dir / f"{split}_metrics_by_cluster.csv")
        if "by_ethnicity_id_a" in metrics:
            save_dataframe(metrics["by_ethnicity_id_a"], method_dir / f"{split}_metrics_by_ethnicity.csv")
        if "by_gender_a" in metrics:
            save_dataframe(metrics["by_gender_a"], method_dir / f"{split}_metrics_by_gender.csv")


def _minimum_sum_method_name(fpr_weight: float, tie_break: str) -> str:
    suffixes = []
    if fpr_weight == 1:
        name = "minimum_sum"
    else:
        weight_text = f"{fpr_weight:g}".replace(".", "p")
        name = f"minimum_sum_fpr{weight_text}"
    if tie_break != "median":
        suffixes.append(tie_break)
    if suffixes:
        return f"{name}_{'_'.join(suffixes)}"
    return name


def create_pair_distance_experiment(
    config: ExperimentConfig,
    pair_config: PairGenerationConfig | None = None,
    threshold_config: ThresholdTrainingConfig | None = None,
    shared_val_pairs_path: Path | None = None,
    minimum_sum_fpr_weights: tuple[float, ...] = (1.0,),
    minimum_sum_tie_breaks: tuple[str, ...] = ("median",),
    distance_metric: str = "cosine",
) -> Path:
    """Generate pairs/distances, then run fixed and minimum-sum threshold evaluations."""

    paths = ProjectPaths(config.project_root, config.dataset_name)
    cluster_dir = paths.clusters_dir(config.cluster_embedding, config.cluster_run_name)
    verification_dir = paths.embeddings_dir("verification", config.verification_embedding)
    pairs_dir = ensure_dir(paths.pairs_dir(config.cluster_embedding, config.cluster_run_name))
    experiment_dir = ensure_dir(
        paths.experiment_dir(
            config.cluster_embedding,
            config.verification_embedding,
            config.cluster_run_name,
        )
    )

    base_pair_config = pair_config or PairGenerationConfig()
    train_pair_generator = PairGenerator(replace(base_pair_config, per_cluster=True))
    val_pair_generator = PairGenerator(replace(base_pair_config, per_cluster=False))
    train_clusters = pd.read_csv(cluster_dir / "train_clusters.csv", dtype=str)
    val_clusters = pd.read_csv(cluster_dir / "val_assigned_clusters.csv", dtype=str)

    train_same, train_diff = train_pair_generator.generate(train_clusters)
    if shared_val_pairs_path is None:
        val_same, val_diff = val_pair_generator.generate(val_clusters)
    else:
        val_same, val_diff = _shared_validation_pairs(shared_val_pairs_path, val_clusters)
    save_dataframe(train_same, pairs_dir / "train_pairs_same.csv")
    save_dataframe(train_diff, pairs_dir / "train_pairs_diff.csv")
    save_dataframe(val_same, pairs_dir / "val_pairs_same.csv")
    save_dataframe(val_diff, pairs_dir / "val_pairs_diff.csv")

    pair_summary = pd.DataFrame(
        [
            {
                "split": "train",
                "same_person_pairs": len(train_same),
                "different_person_pairs": len(train_diff),
                "balanced": len(train_same) == len(train_diff),
                "same_cross_cluster_pairs": int(
                    (train_same["cluster_a"] != train_same["cluster_b"]).sum()
                ),
                "different_cross_cluster_pairs": int(
                    (train_diff["cluster_a"] != train_diff["cluster_b"]).sum()
                ),
            },
            {
                "split": "val",
                "same_person_pairs": len(val_same),
                "different_person_pairs": len(val_diff),
                "balanced": len(val_same) == len(val_diff),
                "same_cross_cluster_pairs": int(
                    (val_same["cluster_a"] != val_same["cluster_b"]).sum()
                ),
                "different_cross_cluster_pairs": int(
                    (val_diff["cluster_a"] != val_diff["cluster_b"]).sum()
                ),
            },
        ]
    )
    save_dataframe(pair_summary, experiment_dir / "pair_generation_summary.csv")
    print(pair_summary.to_string(index=False))

    train_pairs = pd.concat([train_same, train_diff], ignore_index=True)
    val_pairs = pd.concat([val_same, val_diff], ignore_index=True)
    train_calculator = PairDistanceCalculator.from_embedding_dir(
        verification_dir,
        "train",
        verification_embedding_name=config.verification_embedding,
        distance_metric=distance_metric,
    )
    val_calculator = PairDistanceCalculator.from_embedding_dir(
        verification_dir,
        "val",
        verification_embedding_name=config.verification_embedding,
        distance_metric=distance_metric,
    )
    train_distances = train_calculator.compute(train_pairs)
    val_distances = val_calculator.compute(val_pairs)
    save_dataframe(train_distances, experiment_dir / "train_pairs_with_distances.csv")
    save_dataframe(val_distances, experiment_dir / "val_pairs_with_distances.csv")

    cfg = threshold_config or ThresholdTrainingConfig()
    distance_col = train_calculator.distance_col
    cfg = replace(cfg, distance_col=distance_col)
    fixed_thresholds = build_fixed_thresholds_by_group(train_distances, cfg)

    for fpr_weight in minimum_sum_fpr_weights:
        for tie_break in minimum_sum_tie_breaks:
            weighted_cfg = replace(
                cfg,
                fpr_weight=float(fpr_weight),
                tie_break=tie_break,
            )
            minimum_sum_thresholds = train_minimum_sum_thresholds_by_group(
                train_distances,
                weighted_cfg,
            )
            _save_method_outputs(
                experiment_dir,
                _minimum_sum_method_name(float(fpr_weight), tie_break),
                minimum_sum_thresholds,
                train_distances,
                val_distances,
                distance_col,
            )
    _save_method_outputs(
        experiment_dir,
        "fixed",
        fixed_thresholds,
        train_distances,
        val_distances,
        distance_col,
    )

    return experiment_dir
