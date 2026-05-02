"""Witnesses for sparse-encode precompute-dispatch helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_sparse_encode_dispatched_gram(
    gram: AbstractArray | None,
    dictionary: AbstractArray,
    algorithm: str,
) -> AbstractArray | None:
    """Describe the dispatched Gram matrix."""
    del dictionary, algorithm
    if gram is None:
        return None
    return AbstractArray(shape=gram.shape, dtype=gram.dtype)


def witness_sparse_encode_dispatched_covariance(
    cov: AbstractArray | None,
    X: AbstractArray,
    dictionary: AbstractArray,
    algorithm: str,
) -> AbstractArray | None:
    """Describe the dispatched covariance matrix."""
    del X, dictionary, algorithm
    if cov is None:
        return None
    return AbstractArray(shape=cov.shape, dtype=cov.dtype)


def witness_sparse_encode_resolved_copy_cov(
    copy_cov: bool,
    cov: AbstractArray | None,
    algorithm: str,
) -> AbstractArray:
    """Describe the final copy_cov mode after covariance dispatch."""
    del copy_cov, cov, algorithm
    return AbstractArray(shape=(), dtype="bool")
