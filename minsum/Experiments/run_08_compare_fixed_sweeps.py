"""Compare fixed-threshold sweeps against existing minimum-sum results."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from minsum.Core.artifact_store import ensure_dir, save_dataframe
from minsum.Core.paths import ProjectPaths
from minsum.Experiments.run_utils import parse_csv_list
from minsum.Postprocessing.evaluation.evaluator import apply_thresholds, evaluate_pairs


@dataclass(frozen=True)
class RunSpec:
    dataset_name: str
    verification_embedding: str


VERIFIER_LABELS = {
    "arcface_prealigned": "ArcFace",
    "dface_deepface": "DeepFace",
    "magface_iresnet50": "MagFace",
    "transface_vit_s": "TransFace",
}


def _parse_thresholds(raw: str) -> list[float]:
    values = []
    for item in parse_csv_list(raw):
        values.append(float(item))
    if not values:
        raise ValueError("--fixed-thresholds must contain at least one value")
    return values


def _metric_is_higher_better(metric: str) -> bool:
    return metric not in {"fpr", "fnr", "far", "fp", "fn"}


def _experiment_dir(
    project_root: Path,
    dataset_name: str,
    cluster_embedding: str,
    verification_embedding: str,
    cluster_run_name: str,
) -> Path:
    return ProjectPaths(project_root, dataset_name).experiment_dir(
        cluster_embedding,
        verification_embedding,
        cluster_run_name,
    )


def _uniform_thresholds(val_distances: pd.DataFrame, threshold: float) -> pd.DataFrame:
    clusters = sorted(val_distances["cluster_a"].astype(str).dropna().unique().tolist())
    return pd.DataFrame({"cluster": clusters, "threshold": float(threshold)})


def _evaluate_fixed_threshold(
    val_distances: pd.DataFrame,
    threshold: float,
    distance_col: str,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    thresholds = _uniform_thresholds(val_distances, threshold)
    distances = val_distances.copy()
    distances["cluster_a"] = distances["cluster_a"].astype(str)
    classified = apply_thresholds(distances, thresholds, distance_col=distance_col)
    return classified, evaluate_pairs(classified)


def _first_row(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Empty metrics file: {path}")
    return frame.iloc[0].to_dict()


def _load_minsum_global(experiment_dir: Path) -> dict[str, object]:
    return _first_row(experiment_dir / "minimum_sum" / "val_metrics_global.csv")


def _load_minsum_ethnicity(experiment_dir: Path) -> pd.DataFrame | None:
    path = experiment_dir / "minimum_sum" / "val_metrics_by_ethnicity.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path)


def _format_percent(value: object) -> float:
    return round(float(value) * 100, 2)


def _error_reduction(fixed_accuracy: float, minsum_accuracy: float) -> float:
    fixed_error = 1.0 - fixed_accuracy
    minsum_error = 1.0 - minsum_accuracy
    if fixed_error <= 0:
        return 0.0
    return round(((fixed_error - minsum_error) / fixed_error) * 100, 2)


def _global_row(
    dataset_name: str,
    verification_embedding: str,
    method: str,
    threshold: str | float,
    metrics: dict[str, object],
    best_fixed_accuracy: float | None = None,
) -> dict[str, object]:
    accuracy = float(metrics["accuracy"])
    row = {
        "dataset": dataset_name.upper() if dataset_name == "rfw" else "BUPT",
        "verifier": VERIFIER_LABELS.get(verification_embedding, verification_embedding),
        "verification_embedding": verification_embedding,
        "method": method,
        "fixed_tau": threshold,
        "accuracy_%": _format_percent(metrics["accuracy"]),
        "fpr_%": _format_percent(metrics["fpr"]),
        "fnr_%": _format_percent(metrics["fnr"]),
        "precision_%": _format_percent(metrics["precision"]),
        "recall_%": _format_percent(metrics["recall"]),
        "f1_%": _format_percent(metrics["f1"]),
        "fp": int(metrics["fp"]),
        "fn": int(metrics["fn"]),
        "num_pairs": int(metrics["num_pairs"]),
    }
    if method == "minimum_sum" and best_fixed_accuracy is not None:
        row["error_reduction_vs_best_fixed_%"] = _error_reduction(
            best_fixed_accuracy,
            accuracy,
        )
    else:
        row["error_reduction_vs_best_fixed_%"] = ""
    return row


def _disparity_row(
    verification_embedding: str,
    method: str,
    threshold: str | float,
    by_ethnicity: pd.DataFrame,
) -> dict[str, object]:
    return {
        "verifier": VERIFIER_LABELS.get(verification_embedding, verification_embedding),
        "verification_embedding": verification_embedding,
        "method": method,
        "fixed_tau": threshold,
        "fpr_gap_pp": round((by_ethnicity["fpr"].max() - by_ethnicity["fpr"].min()) * 100, 2),
        "fnr_gap_pp": round((by_ethnicity["fnr"].max() - by_ethnicity["fnr"].min()) * 100, 2),
        "f1_gap_pp": round((by_ethnicity["f1"].max() - by_ethnicity["f1"].min()) * 100, 2),
        "worst_fnr_%": round(by_ethnicity["fnr"].max() * 100, 2),
        "worst_f1_%": round(by_ethnicity["f1"].min() * 100, 2),
        "worst_fnr_ethnicity": by_ethnicity.loc[
            by_ethnicity["fnr"].idxmax(),
            "ethnicity_id_a",
        ],
        "worst_f1_ethnicity": by_ethnicity.loc[
            by_ethnicity["f1"].idxmin(),
            "ethnicity_id_a",
        ],
    }


def _run_specs(dataset_names: list[str], verification_embeddings: list[str]) -> list[RunSpec]:
    return [
        RunSpec(dataset_name=dataset_name, verification_embedding=verification_embedding)
        for dataset_name in dataset_names
        for verification_embedding in verification_embeddings
    ]


def _select_shared_thresholds_by_verifier(
    run_evaluations: list[dict[str, object]],
    metric: str,
    higher_better: bool,
) -> dict[str, float]:
    rows = []
    for run in run_evaluations:
        spec = run["spec"]
        assert isinstance(spec, RunSpec)
        for candidate in run["fixed_candidates"]:
            rows.append(
                {
                    "verification_embedding": spec.verification_embedding,
                    "threshold": float(candidate["threshold"]),
                    "metric": float(candidate["global"][metric]),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {}

    grouped = (
        frame.groupby(["verification_embedding", "threshold"], as_index=False)["metric"]
        .mean()
        .rename(columns={"metric": f"mean_{metric}"})
    )
    selected = {}
    metric_col = f"mean_{metric}"
    for verification_embedding, group in grouped.groupby("verification_embedding"):
        ordered = group.sort_values(metric_col, ascending=not higher_better)
        selected[verification_embedding] = float(ordered.iloc[0]["threshold"])
    return selected


def _candidate_by_threshold(
    fixed_candidates: list[dict[str, object]],
    threshold: float,
) -> dict[str, object]:
    for candidate in fixed_candidates:
        if float(candidate["threshold"]) == float(threshold):
            return candidate
    raise ValueError(f"No fixed candidate found for threshold {threshold}")


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--project-root", type=Path, required=True)
    cli.add_argument("--datasets", default="rfw,bupt_cbface50")
    cli.add_argument("--cluster-embedding", default="fairface_penultimate")
    cli.add_argument(
        "--verification-embeddings",
        default="arcface_prealigned,transface_vit_s,magface_iresnet50",
    )
    cli.add_argument("--cluster-run-name", default="k30_cosine_centroid_no_pca")
    cli.add_argument("--distance-col", default="cosine_distance")
    cli.add_argument(
        "--fixed-thresholds",
        default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90",
    )
    cli.add_argument(
        "--best-fixed-metric",
        default="accuracy",
        help="Metric used to choose the best fixed threshold. Default: accuracy.",
    )
    cli.add_argument(
        "--fixed-selection",
        choices=("shared_by_verifier", "per_dataset"),
        default="shared_by_verifier",
        help=(
            "shared_by_verifier chooses one fixed threshold per verifier across "
            "all datasets; per_dataset is an oracle ablation."
        ),
    )
    cli.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: PROJECT_ROOT/outputs/experiments/final_threshold_comparison",
    )
    cli.add_argument("--skip-missing", action="store_true")
    args = cli.parse_args()

    dataset_names = parse_csv_list(args.datasets)
    verification_embeddings = parse_csv_list(args.verification_embeddings)
    fixed_thresholds = _parse_thresholds(args.fixed_thresholds)
    output_dir = ensure_dir(
        args.output_dir
        or args.project_root / "outputs" / "experiments" / "final_threshold_comparison"
    )

    sweep_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    run_evaluations: list[dict[str, object]] = []

    higher_better = _metric_is_higher_better(args.best_fixed_metric)

    for spec in _run_specs(dataset_names, verification_embeddings):
        experiment_dir = _experiment_dir(
            args.project_root,
            spec.dataset_name,
            args.cluster_embedding,
            spec.verification_embedding,
            args.cluster_run_name,
        )
        distances_path = experiment_dir / "val_pairs_with_distances.csv"
        minsum_global_path = experiment_dir / "minimum_sum" / "val_metrics_global.csv"
        if not distances_path.is_file() or not minsum_global_path.is_file():
            missing_rows.append(
                {
                    "dataset_name": spec.dataset_name,
                    "verification_embedding": spec.verification_embedding,
                    "experiment_dir": str(experiment_dir),
                    "missing_val_distances": not distances_path.is_file(),
                    "missing_minsum_global": not minsum_global_path.is_file(),
                }
            )
            if args.skip_missing:
                continue
            raise FileNotFoundError(f"Missing experiment outputs under {experiment_dir}")

        val_distances = pd.read_csv(distances_path)
        if args.distance_col not in val_distances.columns:
            raise ValueError(f"{distances_path} has no {args.distance_col!r} column")

        fixed_candidates = []
        for threshold in fixed_thresholds:
            _, metrics = _evaluate_fixed_threshold(
                val_distances,
                threshold,
                distance_col=args.distance_col,
            )
            global_metrics = metrics["global"].iloc[0].to_dict()
            fixed_candidates.append(
                {
                    "threshold": threshold,
                    "global": global_metrics,
                    "by_ethnicity": metrics.get("by_ethnicity_id_a"),
                }
            )
            sweep_row = _global_row(
                spec.dataset_name,
                spec.verification_embedding,
                method="fixed_sweep",
                threshold=threshold,
                metrics=global_metrics,
            )
            sweep_rows.append(sweep_row)

        run_evaluations.append(
            {
                "spec": spec,
                "experiment_dir": experiment_dir,
                "fixed_candidates": fixed_candidates,
                "minsum_global": _load_minsum_global(experiment_dir),
                "minsum_ethnicity": _load_minsum_ethnicity(experiment_dir),
            }
        )

    if args.fixed_selection == "shared_by_verifier":
        selected_thresholds = _select_shared_thresholds_by_verifier(
            run_evaluations,
            metric=args.best_fixed_metric,
            higher_better=higher_better,
        )
        fixed_method_name = f"shared_best_fixed_by_{args.best_fixed_metric}"
    else:
        selected_thresholds = {}
        fixed_method_name = f"dataset_best_fixed_by_{args.best_fixed_metric}"

    global_rows: list[dict[str, object]] = []
    disparity_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []

    for run in run_evaluations:
        spec = run["spec"]
        assert isinstance(spec, RunSpec)
        fixed_candidates = run["fixed_candidates"]
        if args.fixed_selection == "shared_by_verifier":
            best_fixed_threshold = selected_thresholds[spec.verification_embedding]
            best_fixed = _candidate_by_threshold(fixed_candidates, best_fixed_threshold)
        else:
            best_fixed = sorted(
                fixed_candidates,
                key=lambda item: float(item["global"][args.best_fixed_metric]),
                reverse=higher_better,
            )[0]
            best_fixed_threshold = float(best_fixed["threshold"])

        best_fixed_global = best_fixed["global"]
        minsum_global = run["minsum_global"]
        selected_rows.append(
            {
                "dataset_name": spec.dataset_name,
                "verification_embedding": spec.verification_embedding,
                "verifier": VERIFIER_LABELS.get(
                    spec.verification_embedding,
                    spec.verification_embedding,
                ),
                "fixed_selection": args.fixed_selection,
                "selected_threshold": best_fixed_threshold,
                f"selected_{args.best_fixed_metric}": float(
                    best_fixed_global[args.best_fixed_metric]
                ),
            }
        )

        global_rows.append(
            _global_row(
                spec.dataset_name,
                spec.verification_embedding,
                method=fixed_method_name,
                threshold=best_fixed_threshold,
                metrics=best_fixed_global,
            )
        )
        global_rows.append(
            _global_row(
                spec.dataset_name,
                spec.verification_embedding,
                method="minimum_sum",
                threshold="-",
                metrics=minsum_global,
                best_fixed_accuracy=float(best_fixed_global["accuracy"]),
            )
        )

        if spec.dataset_name == "rfw":
            best_fixed_ethnicity = best_fixed["by_ethnicity"]
            if best_fixed_ethnicity is not None:
                disparity_rows.append(
                    _disparity_row(
                        spec.verification_embedding,
                        method=fixed_method_name,
                        threshold=best_fixed_threshold,
                        by_ethnicity=best_fixed_ethnicity,
                    )
                )
            minsum_ethnicity = run["minsum_ethnicity"]
            if minsum_ethnicity is not None:
                disparity_rows.append(
                    _disparity_row(
                        spec.verification_embedding,
                        method="minimum_sum",
                        threshold="-",
                        by_ethnicity=minsum_ethnicity,
                    )
                )

    global_table = pd.DataFrame(global_rows)
    sweep_table = pd.DataFrame(sweep_rows)
    disparity_table = pd.DataFrame(disparity_rows)
    missing_table = pd.DataFrame(missing_rows)
    selected_table = pd.DataFrame(selected_rows)

    save_dataframe(sweep_table, output_dir / "fixed_threshold_sweep_global.csv")
    save_dataframe(selected_table, output_dir / "selected_fixed_thresholds.csv")
    save_dataframe(global_table, output_dir / "table_best_fixed_vs_minsum_global.csv")
    save_dataframe(disparity_table, output_dir / "table_rfw_best_fixed_vs_minsum_disparity.csv")
    if not missing_table.empty:
        save_dataframe(missing_table, output_dir / "missing_runs.csv")

    print("\nTABLE 1 - BEST FIXED VS MINSUM GLOBAL")
    print(global_table.to_string(index=False))
    if not disparity_table.empty:
        print("\nTABLE 2 - RFW BEST FIXED VS MINSUM DISPARITY")
        print(disparity_table.to_string(index=False))
    if not missing_table.empty:
        print("\nMISSING RUNS")
        print(missing_table.to_string(index=False))
    print(f"\nSaved under: {output_dir}")


if __name__ == "__main__":
    main()
