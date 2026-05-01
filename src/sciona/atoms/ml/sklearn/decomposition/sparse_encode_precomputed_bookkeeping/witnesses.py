"""Ghost witnesses for sparse-encode precomputed bookkeeping atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_sparse_encode_lasso_alpha(
    regularization: float,
    n_features: int,
) -> AbstractArray:
    """Describe lasso alpha scaling inside _sparse_encode_precomputed."""
    del regularization, n_features
    return AbstractArray(shape=(), dtype="float64")


def witness_sparse_encode_writable_init(init: AbstractArray) -> AbstractArray:
    """Describe the writable lasso_cd init buffer."""
    return AbstractArray(shape=init.shape, dtype=init.dtype)


def witness_sparse_encode_omp_norms_squared(X: AbstractArray) -> AbstractArray:
    """Describe the squared row-norm vector for OMP."""
    if len(X.shape) != 2:
        raise ValueError("X must be rank 2")
    return AbstractArray(shape=(X.shape[0],), dtype="float64")


def witness_sparse_encode_precomputed_output(
    new_code: AbstractArray,
    n_samples: int,
    n_components: int,
) -> AbstractArray:
    """Describe the reshaped dense sparse_encode result."""
    del new_code
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if n_components < 1:
        raise ValueError("n_components must be positive")
    return AbstractArray(shape=(n_samples, n_components), dtype="float64")
