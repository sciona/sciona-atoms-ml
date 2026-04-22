"""Ghost witnesses for sklearn spectral-clustering label assignment atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_embedding(vectors: AbstractArray) -> tuple[int, int]:
    if len(vectors.shape) != 2:
        raise ValueError("vectors must be 2D")
    n_samples, n_components = int(vectors.shape[0]), int(vectors.shape[1])
    if n_samples < 1 or n_components < 1:
        raise ValueError("vectors must be nonempty")
    if n_samples < n_components:
        raise ValueError("sample count must be at least component count")
    return n_samples, n_components


def witness_spectral_cluster_qr_labels(vectors: AbstractArray) -> AbstractArray:
    """Describe QR-based labels from a spectral embedding."""
    n_samples, _ = _check_embedding(vectors)
    return AbstractArray(shape=(n_samples,), dtype="int64", min_val=0)


def witness_spectral_discretize_labels(
    vectors: AbstractArray,
    *,
    max_svd_restarts: int = 30,
    n_iter_max: int = 20,
    random_state: int | None = None,
) -> AbstractArray:
    """Describe Yu-Shi discretization labels from a spectral embedding."""
    del random_state
    n_samples, _ = _check_embedding(vectors)
    if max_svd_restarts < 1:
        raise ValueError("max_svd_restarts must be positive")
    if n_iter_max < 1:
        raise ValueError("n_iter_max must be positive")
    return AbstractArray(shape=(n_samples,), dtype="int64", min_val=0)
