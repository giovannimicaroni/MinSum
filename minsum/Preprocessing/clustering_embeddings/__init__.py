"""Embedding extractors used for clustering."""

from .deepface_extractor import DeepFaceConfig, DeepFaceEmbeddingExtractor
from .fairface_extractor import FairFaceConfig, FairFacePenultimateExtractor

__all__ = [
    "DeepFaceConfig",
    "DeepFaceEmbeddingExtractor",
    "FairFaceConfig",
    "FairFacePenultimateExtractor",
]
