"""KMeans training and train/val centroid assignment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from minsum.Core.artifact_store import ensure_dir, save_dataframe, save_json
from .centroid_assignment import (
    assign_by_cosine_to_centroids,
    compute_original_space_centroids,
    l2_normalize,
)
from .cluster_reports import make_cluster_report


@dataclass
class KMeansConfig:
    n_clusters: int = 15
    random_state: int = 42
    use_pca_before_kmeans: bool = False
    pca_components: int | None = None
    kmeans_n_init: int = 10
    kmeans_max_iter: int = 300
    assign_by: str = "cosine_to_original_centroids"


@dataclass
class KMeansResult:
    train_clusters: pd.DataFrame
    val_clusters: pd.DataFrame | None
    original_space_centroids: np.ndarray
    kmeans_model: KMeans
    pca_model: PCA | None
    train_assignment_agreement: float


class KMeansClusterer:
    """Fit KMeans on train embeddings and assign splits by cosine to original centroids."""

    def __init__(self, config: KMeansConfig | None = None) -> None:
        self.config = config or KMeansConfig()

    def fit_assign(
        self,
        train_embeddings: np.ndarray,
        train_metadata: pd.DataFrame,
        val_embeddings: np.ndarray | None = None,
        val_metadata: pd.DataFrame | None = None,
        output_dir: Path | None = None,
    ) -> KMeansResult:
        cfg = self.config
        train_l2 = l2_normalize(train_embeddings)
        training_space = train_l2
        pca_model = None
        if cfg.use_pca_before_kmeans:
            if cfg.pca_components is None:
                raise ValueError("pca_components must be set when use_pca_before_kmeans=True")
            pca_model = PCA(n_components=cfg.pca_components, random_state=cfg.random_state)
            training_space = l2_normalize(pca_model.fit_transform(train_l2))

        kmeans = KMeans(
            n_clusters=cfg.n_clusters,
            n_init=cfg.kmeans_n_init,
            max_iter=cfg.kmeans_max_iter,
            random_state=cfg.random_state,
        )
        train_kmeans_labels = kmeans.fit_predict(training_space).astype(int)
        original_centroids = compute_original_space_centroids(
            train_l2,
            train_kmeans_labels,
            cfg.n_clusters,
        )
        train_cosine_labels, train_sim, train_dist = assign_by_cosine_to_centroids(
            train_embeddings,
            original_centroids,
        )
        agreement = float((train_kmeans_labels == train_cosine_labels).mean())

        train_clusters = self._cluster_frame(
            train_metadata,
            cluster=train_kmeans_labels,
            similarity=train_sim,
            distance=train_dist,
        )
        train_clusters["cluster_by_cosine_original_centroid"] = train_cosine_labels.astype(int)

        val_clusters = None
        if val_embeddings is not None and val_metadata is not None:
            val_labels, val_sim, val_dist = assign_by_cosine_to_centroids(
                val_embeddings,
                original_centroids,
            )
            val_clusters = self._cluster_frame(val_metadata, val_labels, val_sim, val_dist)

        result = KMeansResult(
            train_clusters=train_clusters,
            val_clusters=val_clusters,
            original_space_centroids=original_centroids,
            kmeans_model=kmeans,
            pca_model=pca_model,
            train_assignment_agreement=agreement,
        )
        if output_dir is not None:
            self.save_result(result, Path(output_dir))
        return result

    @staticmethod
    def _cluster_frame(
        metadata: pd.DataFrame,
        cluster: np.ndarray,
        similarity: np.ndarray,
        distance: np.ndarray,
    ) -> pd.DataFrame:
        frame = metadata.copy().reset_index(drop=True)
        frame = frame.drop(columns=["index"], errors="ignore")
        frame.insert(0, "index", range(len(frame)))
        frame["cluster"] = cluster.astype(int)
        frame["similarity_to_centroid"] = similarity.astype(float)
        frame["cosine_distance_to_centroid"] = distance.astype(float)
        return frame

    def save_result(self, result: KMeansResult, output_dir: Path) -> None:
        output_dir = ensure_dir(output_dir)
        joblib.dump(result.kmeans_model, output_dir / "kmeans_model.pkl")
        if result.pca_model is not None:
            joblib.dump(result.pca_model, output_dir / "pca_model.pkl")
        np.save(output_dir / "kmeans_centroids_space.npy", result.kmeans_model.cluster_centers_.astype(np.float32))
        np.save(output_dir / "original_space_centroids.npy", result.original_space_centroids.astype(np.float32))
        centroids_df = pd.DataFrame(result.original_space_centroids)
        centroids_df.insert(0, "cluster", range(result.original_space_centroids.shape[0]))
        save_dataframe(centroids_df, output_dir / "original_space_centroids.csv")
        save_dataframe(result.train_clusters, output_dir / "train_clusters.csv")
        save_dataframe(make_cluster_report(result.train_clusters, "train"), output_dir / "cluster_report_train.csv")
        generated_files = {
            "kmeans_model": str(output_dir / "kmeans_model.pkl"),
            "pca_model": str(output_dir / "pca_model.pkl") if result.pca_model is not None else None,
            "kmeans_centroids_space": str(output_dir / "kmeans_centroids_space.npy"),
            "original_space_centroids": str(output_dir / "original_space_centroids.npy"),
            "original_space_centroids_csv": str(output_dir / "original_space_centroids.csv"),
            "train_clusters": str(output_dir / "train_clusters.csv"),
            "cluster_report_train": str(output_dir / "cluster_report_train.csv"),
        }
        if result.val_clusters is not None:
            save_dataframe(result.val_clusters, output_dir / "val_assigned_clusters.csv")
            save_dataframe(make_cluster_report(result.val_clusters, "val"), output_dir / "cluster_report_val.csv")
            generated_files["val_assigned_clusters"] = str(output_dir / "val_assigned_clusters.csv")
            generated_files["cluster_report_val"] = str(output_dir / "cluster_report_val.csv")

        config = asdict(self.config)
        config.update(
            {
                "train_kmeans_vs_original_centroid_agreement": result.train_assignment_agreement,
                "kmeans_inertia": float(result.kmeans_model.inertia_),
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "generated_files": generated_files,
            }
        )
        save_json(config, output_dir / "config.json")
