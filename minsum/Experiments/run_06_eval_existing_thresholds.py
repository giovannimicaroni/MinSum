"""Evaluate existing thresholds on a newly generated or shared validation set."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from minsum.Core.artifact_store import ensure_dir, save_dataframe
from minsum.Core.paths import ProjectPaths
from minsum.Experiments.pipeline import _shared_validation_pairs
from minsum.Experiments.run_utils import parse_csv_list
from minsum.Postprocessing.distances.pair_distance_calculator import (
    PairDistanceCalculator,
)
from minsum.Postprocessing.evaluation.evaluator import apply_thresholds, evaluate_pairs
from minsum.Postprocessing.pairs.pair_generator import (
    PairGenerationConfig,
    PairGenerator,
)


def _save_eval_outputs(
    output_dir: Path,
    method_name: str,
    thresholds: pd.DataFrame,
    val_distances: pd.DataFrame,
    distance_col: str,
) -> None:
    method_dir = ensure_dir(output_dir / method_name)
    thresholds = thresholds.copy()
    val_distances = val_distances.copy()
    thresholds["cluster"] = thresholds["cluster"].astype(str)
    if "cluster_a" in val_distances.columns:
        val_distances["cluster_a"] = val_distances["cluster_a"].astype(str)
    save_dataframe(thresholds, method_dir / "thresholds_by_cluster.csv")

    classified = apply_thresholds(val_distances, thresholds, distance_col=distance_col)
    save_dataframe(classified, method_dir / "val_pairs_classified.csv")

    metrics = evaluate_pairs(classified)
    save_dataframe(metrics["global"], method_dir / "val_metrics_global.csv")
    if "by_cluster_a" in metrics:
        save_dataframe(metrics["by_cluster_a"], method_dir / "val_metrics_by_cluster.csv")
    if "by_ethnicity_id_a" in metrics:
        save_dataframe(metrics["by_ethnicity_id_a"], method_dir / "val_metrics_by_ethnicity.csv")
    if "by_gender_a" in metrics:
        save_dataframe(metrics["by_gender_a"], method_dir / "val_metrics_by_gender.csv")


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--project-root", type=Path, required=True)
    cli.add_argument("--dataset-name", default="rfw")
    cli.add_argument("--cluster-embedding", required=True)
    cli.add_argument("--verification-embedding", required=True)
    cli.add_argument("--cluster-run-name", default="k30_cosine_centroid_no_pca")
    cli.add_argument("--random-seed", type=int, default=42)
    cli.add_argument("--pairs-per-class", type=int, default=10000)
    cli.add_argument("--allow-different-ethnicity", action="store_true")
    cli.add_argument("--diff-constraint-cols", default="ethnicity_id")
    cli.add_argument("--distance-metric", choices=("cosine", "euclidean"), default="cosine")
    cli.add_argument(
        "--threshold-methods",
        default="fixed,minimum_sum",
        help="Comma-separated existing method directories to evaluate.",
    )
    cli.add_argument(
        "--eval-name",
        default=None,
        help="Output directory name under the experiment directory. Defaults to val_p{pairs-per-class}_eval.",
    )
    cli.add_argument(
        "--shared-val-pairs-path",
        type=Path,
        default=None,
        help="Optional validation pair CSV generated once and reused across clustering embeddings.",
    )
    cli.add_argument(
        "--write-shared-val-pairs-path",
        type=Path,
        default=None,
        help="Optional path where this run writes the generated validation pair CSV for reuse.",
    )
    args = cli.parse_args()

    paths = ProjectPaths(args.project_root, args.dataset_name)
    cluster_dir = paths.clusters_dir(args.cluster_embedding, args.cluster_run_name)
    verification_dir = paths.embeddings_dir("verification", args.verification_embedding)
    experiment_dir = paths.experiment_dir(
        args.cluster_embedding,
        args.verification_embedding,
        args.cluster_run_name,
    )
    eval_name = args.eval_name or f"val_p{args.pairs_per_class}_eval"
    output_dir = ensure_dir(experiment_dir / eval_name)

    val_clusters = pd.read_csv(cluster_dir / "val_assigned_clusters.csv", dtype=str)
    pair_config = PairGenerationConfig(
        per_cluster=False,
        same_ethnicity_for_diff=not args.allow_different_ethnicity,
        max_pairs_per_cluster=args.pairs_per_class,
        random_seed=args.random_seed,
        diff_constraint_cols=parse_csv_list(args.diff_constraint_cols),
    )
    if args.shared_val_pairs_path is None:
        val_same, val_diff = PairGenerator(pair_config).generate(val_clusters)
    else:
        val_same, val_diff = _shared_validation_pairs(args.shared_val_pairs_path, val_clusters)

    save_dataframe(val_same, output_dir / "val_pairs_same.csv")
    save_dataframe(val_diff, output_dir / "val_pairs_diff.csv")
    if args.write_shared_val_pairs_path is not None:
        ensure_dir(args.write_shared_val_pairs_path.parent)
        save_dataframe(
            pd.concat([val_same, val_diff], ignore_index=True),
            args.write_shared_val_pairs_path,
        )

    pair_summary = pd.DataFrame(
        [
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
            }
        ]
    )
    save_dataframe(pair_summary, output_dir / "pair_generation_summary.csv")
    print(pair_summary.to_string(index=False))

    val_pairs = pd.concat([val_same, val_diff], ignore_index=True)
    calculator = PairDistanceCalculator.from_embedding_dir(
        verification_dir,
        "val",
        verification_embedding_name=args.verification_embedding,
        distance_metric=args.distance_metric,
    )
    val_distances = calculator.compute(val_pairs)
    save_dataframe(val_distances, output_dir / "val_pairs_with_distances.csv")

    methods = parse_csv_list(args.threshold_methods)
    if not methods:
        raise ValueError("--threshold-methods must contain at least one method")
    for method in methods:
        thresholds_path = experiment_dir / method / "thresholds_by_cluster.csv"
        if not thresholds_path.is_file():
            raise FileNotFoundError(f"Existing thresholds not found: {thresholds_path}")
        thresholds = pd.read_csv(thresholds_path)
        _save_eval_outputs(
            output_dir,
            method,
            thresholds,
            val_distances,
            distance_col=calculator.distance_col,
        )

    print(f"Eval-only outputs saved to: {output_dir}")
    if args.write_shared_val_pairs_path is not None:
        print(f"Shared validation pairs saved to: {args.write_shared_val_pairs_path}")


if __name__ == "__main__":
    main()
