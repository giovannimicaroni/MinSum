"""Threshold selection methods for verification distances."""

from __future__ import annotations

import numpy as np


def confusion_counts(labels: np.ndarray, pred_same: np.ndarray) -> dict[str, int]:
    labels = labels.astype(int)
    pred_same = pred_same.astype(int)
    return {
        "tp": int(((labels == 1) & (pred_same == 1)).sum()),
        "tn": int(((labels == 0) & (pred_same == 0)).sum()),
        "fp": int(((labels == 0) & (pred_same == 1)).sum()),
        "fn": int(((labels == 1) & (pred_same == 0)).sum()),
    }


def rates(counts: dict[str, int]) -> dict[str, float]:
    fp, fn, tp, tn = counts["fp"], counts["fn"], counts["tp"], counts["tn"]
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return {"fpr": fpr, "fnr": fnr}


def evaluate_threshold(
    distances: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    fpr_weight: float = 1.0,
) -> dict[str, float | int]:
    pred_same = distances <= threshold
    counts = confusion_counts(labels, pred_same)
    computed_rates = rates(counts)
    return {
        **counts,
        **computed_rates,
        "objective": (fpr_weight * computed_rates["fpr"]) + computed_rates["fnr"],
    }


def select_fixed_threshold(
    distances: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Evaluate a fixed threshold without optimizing it."""

    return {
        "threshold": float(threshold),
        **evaluate_threshold(distances, labels, float(threshold)),
    }


def select_minimum_sum(
    distances: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
    fpr_weight: float = 1.0,
    tie_break: str = "median",
) -> dict[str, float | int]:
    """Select the median threshold among candidates minimizing weighted FPR + FNR."""
    valid_tie_breaks = {"median", "lowest", "highest"}
    if tie_break not in valid_tie_breaks:
        raise ValueError(f"Unknown tie_break {tie_break!r}; expected one of {sorted(valid_tie_breaks)}")

    candidates = []
    for threshold in thresholds:
        metrics = evaluate_threshold(
            distances,
            labels,
            float(threshold),
            fpr_weight=fpr_weight,
        )
        candidates.append({"threshold": float(threshold), **metrics})

    objectives = np.asarray([float(row["objective"]) for row in candidates])
    min_objective = float(objectives.min())
    tied = [
        row
        for row, objective in zip(candidates, objectives)
        if np.isclose(objective, min_objective)
    ]
    tied_thresholds = np.asarray([float(row["threshold"]) for row in tied])
    if tie_break == "lowest":
        selected_threshold = float(tied_thresholds.min())
    elif tie_break == "highest":
        selected_threshold = float(tied_thresholds.max())
    else:
        selected_threshold = float(np.median(tied_thresholds))
    selected_metrics = evaluate_threshold(
        distances,
        labels,
        selected_threshold,
        fpr_weight=fpr_weight,
    )

    return {
        "threshold": selected_threshold,
        **selected_metrics,
        "fpr_weight": float(fpr_weight),
        "tie_break": tie_break,
        "min_objective": min_objective,
        "num_tied_thresholds": int(len(tied_thresholds)),
        "tied_threshold_min": float(tied_thresholds.min()),
        "tied_threshold_max": float(tied_thresholds.max()),
    }
