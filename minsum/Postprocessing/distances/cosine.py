"""Cosine similarity and distance helpers."""

from __future__ import annotations

import numpy as np


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(vector_a, vector_b) / denom)


def cosine_distance(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    similarity = cosine_similarity(vector_a, vector_b)
    if np.isnan(similarity):
        return float("nan")
    return 1.0 - similarity


def euclidean_distance(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    vector_a = np.asarray(vector_a, dtype=np.float32)
    vector_b = np.asarray(vector_b, dtype=np.float32)
    if vector_a.shape != vector_b.shape:
        return float("nan")
    return float(np.linalg.norm(vector_a - vector_b))
