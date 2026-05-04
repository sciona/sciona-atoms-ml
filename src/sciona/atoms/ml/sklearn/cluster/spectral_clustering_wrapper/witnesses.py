"""Ghost witnesses for spectral clustering public-wrapper atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_spectral_clustering_precomputed_affinity(
    parent_affinity: str | None = None,
) -> AbstractArray:
    """Describe the fixed precomputed affinity used by spectral_clustering."""
    del parent_affinity
    return AbstractArray(shape=(), dtype="object")


def witness_spectral_clustering_return_labels(labels: AbstractArray) -> AbstractArray:
    """Describe the labels returned by spectral_clustering."""
    return AbstractArray(shape=labels.shape, dtype="int64")
