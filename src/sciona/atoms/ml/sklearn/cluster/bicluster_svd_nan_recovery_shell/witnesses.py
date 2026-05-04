"""Ghost witnesses for deterministic biclustering SVD fallback setup."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_bicluster_svd_vt_nan_recovery_required(vt: AbstractArray) -> AbstractArray:
    """Describe whether BaseSpectral._svd should repair NaNs in vt."""
    del vt
    return AbstractArray(shape=(), dtype="bool")


def witness_bicluster_svd_u_nan_recovery_required(u: AbstractArray) -> AbstractArray:
    """Describe whether BaseSpectral._svd should repair NaNs in u."""
    del u
    return AbstractArray(shape=(), dtype="bool")


def witness_bicluster_svd_right_gram_matrix(array: AbstractArray) -> AbstractArray:
    """Describe the right-side Gram matrix used for vt NaN recovery."""
    if len(array.shape) != 2:
        raise ValueError("array must be 2D")
    return AbstractArray(
        shape=(int(array.shape[1]), int(array.shape[1])),
        dtype="float64",
    )


def witness_bicluster_svd_left_gram_matrix(array: AbstractArray) -> AbstractArray:
    """Describe the left-side Gram matrix used for u NaN recovery."""
    if len(array.shape) != 2:
        raise ValueError("array must be 2D")
    return AbstractArray(
        shape=(int(array.shape[0]), int(array.shape[0])),
        dtype="float64",
    )


def witness_bicluster_svd_arpack_init_vector(
    random_state: object,
    width: int,
) -> AbstractArray:
    """Describe the ARPACK-style initialization vector used for eigsh fallback."""
    del random_state
    if width < 1:
        raise ValueError("width must be positive")
    return AbstractArray(shape=(width,), dtype="float64")


def witness_bicluster_svd_eigsh_kwargs(
    n_svd_vecs: int | None,
    v0: AbstractArray,
) -> AbstractArray:
    """Describe the kwargs mapping passed into eigsh during NaN recovery."""
    del n_svd_vecs
    del v0
    return AbstractArray(shape=(), dtype="object")
