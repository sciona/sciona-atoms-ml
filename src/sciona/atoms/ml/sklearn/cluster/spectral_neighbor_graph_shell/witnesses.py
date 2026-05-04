"""Ghost witnesses for SpectralClustering neighbor-graph atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_spectral_fit_kneighbors_graph_kwargs(
    n_neighbors: int,
    n_jobs: int | None = None,
) -> AbstractArray:
    """Describe the kwargs passed into kneighbors_graph for nearest-neighbor affinity."""
    del n_neighbors
    del n_jobs
    return AbstractArray(shape=(), dtype="object")


def witness_spectral_fit_precomputed_neighbor_metric(parent_metric: str | None = None) -> AbstractArray:
    """Describe the fixed metric used by the precomputed-neighbor estimator."""
    del parent_metric
    return AbstractArray(shape=(), dtype="object")


def witness_spectral_fit_precomputed_neighbor_estimator_kwargs(
    n_neighbors: int,
    n_jobs: int | None = None,
) -> AbstractArray:
    """Describe the kwargs passed into NearestNeighbors for precomputed-neighbor affinity."""
    del n_neighbors
    del n_jobs
    return AbstractArray(shape=(), dtype="object")


def witness_spectral_fit_precomputed_kneighbors_graph_mode(parent_mode: str | None = None) -> AbstractArray:
    """Describe the fixed mode used by estimator.kneighbors_graph in precomputed-neighbor affinity."""
    del parent_mode
    return AbstractArray(shape=(), dtype="object")
