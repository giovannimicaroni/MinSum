"""Centroid computation and cosine assignment helpers."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import normalize


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    return normalize(np.asarray(matrix), norm="l2", axis=1)


def compute_original_space_centroids(
    embeddings_l2: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    centers = np.zeros((n_clusters, embeddings_l2.shape[1]), dtype=np.float32)
    for cluster in range(n_clusters):
        mask = labels == cluster
        if not np.any(mask):
            continue
        centers[cluster] = embeddings_l2[mask].mean(axis=0)
    return l2_normalize(centers).astype(np.float32)


def assign_by_cosine_to_centroids(
    embeddings: np.ndarray,
    original_space_centroids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    embeddings_l2 = l2_normalize(embeddings)
    centers_l2 = l2_normalize(original_space_centroids)
    similarities = embeddings_l2 @ centers_l2.T
    clusters = np.argmax(similarities, axis=1).astype(int)
    max_similarities = similarities[np.arange(len(embeddings_l2)), clusters].astype(float)
    distances = 1.0 - max_similarities
    return clusters, max_similarities, distances
