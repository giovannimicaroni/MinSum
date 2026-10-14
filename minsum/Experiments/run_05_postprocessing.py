"""Generate pairs, compute distances, and evaluate fixed plus minimum-sum thresholds."""

from __future__ import annotations

import argparse
from pathlib import Path

from minsum.Experiments.configs import ExperimentConfig
from minsum.Experiments.pipeline import create_pair_distance_experiment
from minsum.Experiments.run_utils import parse_csv_list
from minsum.Postprocessing.pairs.pair_generator import PairGenerationConfig
from minsum.Postprocessing.thresholds.threshold_trainer import ThresholdTrainingConfig


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--project-root", type=Path, required=True)
    cli.add_argument("--dataset-name", default="rfw")
    cli.add_argument("--cluster-embedding", required=True)
    cli.add_argument("--verification-embedding", default="arcface")
    cli.add_argument("--cluster-run-name", default="k15_cosine_centroid_no_pca")
    cli.add_argument("--random-seed", type=int, default=42)
    cli.add_argument("--allow-different-ethnicity", action="store_true")
    cli.add_argument(
        "--pairs-per-class",
        "--max-pairs-per-cluster",
        dest="pairs_per_class",
        type=int,
        default=1000,
        help=(
            "Maximum same-person and different-person pairs per train cluster and "
            "across the complete validation split. "
            "The actual counts are always balanced and may be lower when pairs are unavailable."
        ),
    )
    cli.add_argument("--diff-constraint-cols", default="ethnicity_id")
    cli.add_argument(
        "--shared-val-pairs-path",
        type=Path,
        default=None,
        help=(
            "Optional validation pair CSV reused across clustering embeddings. "
            "The images/labels are kept fixed, while cluster columns are reassigned "
            "from the current run's val_assigned_clusters.csv."
        ),
    )
    cli.add_argument("--fixed-threshold", type=float, default=0.45)
    cli.add_argument(
        "--distance-metric",
        choices=("cosine", "euclidean"),
        default="cosine",
        help="Distance metric for verification embeddings. Use euclidean for raw dlib descriptors.",
    )
    cli.add_argument("--num-thresholds", type=int, default=200)
    cli.add_argument(
        "--minimum-sum-fpr-weights",
        default="1",
        help=(
            "Comma-separated FPR weights for weighted minimum-sum objectives. "
            "For example, '1,3,5' writes minimum_sum, minimum_sum_fpr3, "
            "and minimum_sum_fpr5, optimizing weight*FPR + FNR."
        ),
    )
    cli.add_argument(
        "--minimum-sum-tie-breaks",
        default="median",
        help=(
            "Comma-separated tie-break rules for thresholds tied on the minimum-sum "
            "objective. Choices: median, lowest, highest. Lowest is the most "
            "conservative and least likely to create false positives."
        ),
    )
    args = cli.parse_args()
    fpr_weights = tuple(float(value) for value in parse_csv_list(args.minimum_sum_fpr_weights))
    if not fpr_weights:
        raise ValueError("--minimum-sum-fpr-weights must contain at least one value")
    if any(weight <= 0 for weight in fpr_weights):
        raise ValueError("--minimum-sum-fpr-weights values must be positive")
    tie_breaks = tuple(parse_csv_list(args.minimum_sum_tie_breaks))
    valid_tie_breaks = {"median", "lowest", "highest"}
    if not tie_breaks:
        raise ValueError("--minimum-sum-tie-breaks must contain at least one value")
    invalid_tie_breaks = sorted(set(tie_breaks).difference(valid_tie_breaks))
    if invalid_tie_breaks:
        raise ValueError(
            "--minimum-sum-tie-breaks contains invalid values: "
            f"{invalid_tie_breaks}; expected one of {sorted(valid_tie_breaks)}"
        )

    pair_config = PairGenerationConfig(
        per_cluster=True,
        same_ethnicity_for_diff=not args.allow_different_ethnicity,
        max_pairs_per_cluster=args.pairs_per_class,
        random_seed=args.random_seed,
        diff_constraint_cols=parse_csv_list(args.diff_constraint_cols),
    )
    threshold_config = ThresholdTrainingConfig(
        fixed_threshold=args.fixed_threshold,
        num_thresholds=args.num_thresholds,
    )
    experiment_dir = create_pair_distance_experiment(
        ExperimentConfig(
            project_root=args.project_root,
            dataset_name=args.dataset_name,
            cluster_embedding=args.cluster_embedding,
            verification_embedding=args.verification_embedding,
            cluster_run_name=args.cluster_run_name,
            random_seed=args.random_seed,
        ),
        pair_config=pair_config,
        threshold_config=threshold_config,
        shared_val_pairs_path=args.shared_val_pairs_path,
        minimum_sum_fpr_weights=fpr_weights,
        minimum_sum_tie_breaks=tie_breaks,
        distance_metric=args.distance_metric,
    )
    print(f"Experiment saved to: {experiment_dir}")
    print("Threshold outputs saved under: fixed/ and minimum_sum* directories")


if __name__ == "__main__":
    main()
