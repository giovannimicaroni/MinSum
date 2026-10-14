# Modular Experiment Pipeline

This directory is the orchestration layer for the generic refactor.

Notebook mapping:

- `01_build_metadata.ipynb` -> `minsum.Datasets`
- `02_extract_clustering_embeddings_deepface.ipynb` -> `minsum.Preprocessing.clustering_embeddings.deepface_extractor`
- `03_extract_clustering_embeddings_fairface.ipynb` -> `minsum.Preprocessing.clustering_embeddings.fairface_extractor`
- `04_fit_kmeans_and_assign_val_cosine_centroid.ipynb` -> `minsum.Preprocessing.clustering`
- `05_extract_verification_embeddings_arcface_insightface.ipynb` -> `minsum.VerificationEmbeddings.arcface.insightface_extractor`

The notebooks should become thin research interfaces: configure paths, call the
package functions, and display checks. The package owns the reusable logic.
