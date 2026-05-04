"""Ghost witnesses for deterministic biclustering KMeans callback setup."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_bicluster_kmeans_kwargs(
    n_clusters: int,
    init: str | object,
    n_init: int,
    random_state: object,
) -> AbstractArray:
    """Describe the kwargs mapping passed into KMeans by BaseSpectral._k_means."""
    del n_clusters
    del init
    del n_init
    del random_state
    return AbstractArray(shape=(), dtype="object")


def witness_bicluster_minibatch_kmeans_kwargs(
    n_clusters: int,
    init: str | object,
    n_init: int,
    random_state: object,
) -> AbstractArray:
    """Describe the kwargs mapping passed into MiniBatchKMeans by BaseSpectral._k_means."""
    del n_clusters
    del init
    del n_init
    del random_state
    return AbstractArray(shape=(), dtype="object")
