"""Ghost witnesses for deterministic biclustering SVD callback setup."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_bicluster_svd_use_randomized(svd_method: str) -> AbstractArray:
    """Describe whether BaseSpectral._svd uses randomized_svd."""
    del svd_method
    return AbstractArray(shape=(), dtype="bool")


def witness_bicluster_svd_use_arpack(svd_method: str) -> AbstractArray:
    """Describe whether BaseSpectral._svd uses svds/arpack."""
    del svd_method
    return AbstractArray(shape=(), dtype="bool")


def witness_bicluster_svd_randomized_kwargs(
    random_state: object,
    n_svd_vecs: int | None,
) -> AbstractArray:
    """Describe the kwargs mapping passed into randomized_svd."""
    del random_state
    del n_svd_vecs
    return AbstractArray(shape=(), dtype="object")


def witness_bicluster_svd_svds_kwargs(
    n_components: int,
    n_svd_vecs: int | None,
) -> AbstractArray:
    """Describe the kwargs mapping passed into svds."""
    del n_components
    del n_svd_vecs
    return AbstractArray(shape=(), dtype="object")
