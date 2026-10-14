"""Classification metrics for verification pairs."""

from __future__ import annotations

import pandas as pd


def classification_metrics(labels: pd.Series, predictions: pd.Series) -> dict[str, float | int]:
    labels = labels.astype(int)
    predictions = predictions.astype(int)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "fnr": fn / (fn + tp) if (fn + tp) else 0.0,
        "far": fp / (fp + tn) if (fp + tn) else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num_pairs": total,
    }
