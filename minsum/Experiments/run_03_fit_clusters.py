"""Fit KMeans on train embeddings and assign validation by cosine centroid."""

from __future__ import annotations

import argparse
from pathlib import Path

from minsum.Preprocessing.clustering.kmeans_service import KMeansClusterer, KMeansConfig
from minsum.Core.paths import ProjectPaths
from minsum.Core.embedding_store import EmbeddingStore
from minsum.Experiments.run_utils import default_cluster_run_name


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--project-root", type=Path, required=True)
    cli.add_argument("--dataset-name", default="rfw")
    cli.add_argument("--cluster-embedding", default="deepface_arcface")
    cli.add_argument("--embedding-dir", type=Path, default=None)
    cli.add_argument("--output-dir", type=Path, default=None)
    cli.add_argument("--run-name", default=None)
    cli.add_argument("--n-clusters", type=int, default=15)
    cli.add_argument("--random-state", type=int, default=42)
    cli.add_argument("--use-pca", action="store_true")
    cli.add_argument("--pca-components", type=int, default=64)
    cli.add_argument("--kmeans-n-init", type=int, default=10)
    cli.add_argument("--kmeans-max-iter", type=int, default=300)
    args = cli.parse_args()

    paths = ProjectPaths(args.project_root, args.dataset_name)
    embedding_dir = args.embedding_dir or paths.embeddings_dir("clustering", args.cluster_embedding)
    run_name = args.run_name or default_cluster_run_name(
        args.n_clusters,
        args.use_pca,
        args.pca_components if args.use_pca else None,
    )
    output_dir = args.output_dir or paths.clusters_dir(args.cluster_embedding, run_name)

    store = EmbeddingStore(embedding_dir)
    train = store.load_split("train")
    val = store.load_split("val")

    clusterer = KMeansClusterer(
        KMeansConfig(
            n_clusters=args.n_clusters,
            random_state=args.random_state,
            use_pca_before_kmeans=args.use_pca,
            pca_components=args.pca_components if args.use_pca else None,
            kmeans_n_init=args.kmeans_n_init,
            kmeans_max_iter=args.kmeans_max_iter,
        )
    )
    result = clusterer.fit_assign(
        train_embeddings=train.embeddings,
        train_metadata=train.metadata,
        val_embeddings=val.embeddings,
        val_metadata=val.metadata,
        output_dir=output_dir,
    )
    print(f"Clusters saved to: {output_dir}")
    print(f"Train KMeans vs original-centroid agreement: {result.train_assignment_agreement:.6f}")


if __name__ == "__main__":
    main()
