"""Helpers for deterministic biclustering SVD fallback setup adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_bicluster_svd_arpack_init_vector,
    witness_bicluster_svd_eigsh_kwargs,
    witness_bicluster_svd_left_gram_matrix,
    witness_bicluster_svd_right_gram_matrix,
    witness_bicluster_svd_u_nan_recovery_required,
    witness_bicluster_svd_vt_nan_recovery_required,
)

def _finite_or_nan_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1)

def _bool(value: object) -> bool:
    return isinstance(value, bool)

def _finite_dense_or_sparse_matrix(values: object) -> bool:
    if sp.issparse(values):
        return bool(
            values.ndim == 2
            and values.shape[0] >= 1
            and values.shape[1] >= 1
            and np.all(np.isfinite(values.data))
        )
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 2
        and array.shape[0] >= 1
        and array.shape[1] >= 1
        and np.all(np.isfinite(array))
    )

def _same_square_shape(result: object, size: int) -> bool:
    if sp.issparse(result):
        return bool(result.shape == (size, size))
    array = np.asarray(result, dtype=np.float64)
    return bool(array.shape == (size, size) and np.all(np.isfinite(array)))

def _valid_random_state_like(value: object) -> bool:
    return bool(
        value is None
        or isinstance(value, (int, np.integer, np.random.RandomState))
    )

def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)

def _finite_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))

def _optional_positive_int(value: object) -> bool:
    return bool(value is None or _positive_int(value))

def _same_vector_shape_and_bounds(result: object, width: int) -> bool:
    array = np.asarray(result, dtype=np.float64)
    return bool(
        array.shape == (width,)
        and np.all(np.isfinite(array))
        and np.all(array >= -1.0)
        and np.all(array <= 1.0)
    )

def _eigsh_kwargs_valid(result: object, n_svd_vecs: int | None, v0: object) -> bool:
    return bool(
        isinstance(result, dict)
        and set(result) == {"ncv", "v0"}
        and result["ncv"] == n_svd_vecs
        and np.array_equal(np.asarray(result["v0"], dtype=np.float64), np.asarray(v0, dtype=np.float64))
    )

@register_atom(witness_bicluster_svd_vt_nan_recovery_required)
@icontract.require(lambda vt: _finite_or_nan_matrix(vt), "vt must be a nonempty 2D float-like matrix")
@icontract.ensure(lambda result: _bool(result), "result must be boolean")
def bicluster_svd_vt_nan_recovery_required(vt: NDArray[np.float64]) -> bool:
    """Return whether BaseSpectral._svd should repair NaNs in vt."""
    return bool(np.any(np.isnan(np.asarray(vt, dtype=np.float64))))

@register_atom(witness_bicluster_svd_u_nan_recovery_required)
@icontract.require(lambda u: _finite_or_nan_matrix(u), "u must be a nonempty 2D float-like matrix")
@icontract.ensure(lambda result: _bool(result), "result must be boolean")
def bicluster_svd_u_nan_recovery_required(u: NDArray[np.float64]) -> bool:
    """Return whether BaseSpectral._svd should repair NaNs in u."""
    return bool(np.any(np.isnan(np.asarray(u, dtype=np.float64))))

@register_atom(witness_bicluster_svd_right_gram_matrix)
@icontract.require(lambda array: _finite_dense_or_sparse_matrix(array), "array must be a finite nonempty dense or sparse matrix")
@icontract.ensure(
    lambda result, array: _same_square_shape(result, int(np.asarray(array).shape[1] if not sp.issparse(array) else array.shape[1])),
    "result must be a square Gram matrix on the feature axis",
)
def bicluster_svd_right_gram_matrix(array: object) -> object:
    from sklearn.utils.extmath import safe_sparse_dot
    """Build the right-side Gram matrix used for vt NaN recovery."""
    if sp.issparse(array):
        return safe_sparse_dot(array.T, array)
    values = np.asarray(array, dtype=np.float64)
    return safe_sparse_dot(values.T, values)

@register_atom(witness_bicluster_svd_left_gram_matrix)
@icontract.require(lambda array: _finite_dense_or_sparse_matrix(array), "array must be a finite nonempty dense or sparse matrix")
@icontract.ensure(
    lambda result, array: _same_square_shape(result, int(np.asarray(array).shape[0] if not sp.issparse(array) else array.shape[0])),
    "result must be a square Gram matrix on the sample axis",
)
def bicluster_svd_left_gram_matrix(array: object) -> object:
    from sklearn.utils.extmath import safe_sparse_dot
    """Build the left-side Gram matrix used for u NaN recovery."""
    if sp.issparse(array):
        return safe_sparse_dot(array, array.T)
    values = np.asarray(array, dtype=np.float64)
    return safe_sparse_dot(values, values.T)

@register_atom(witness_bicluster_svd_arpack_init_vector)
@icontract.require(lambda random_state: _valid_random_state_like(random_state), "random_state must be None, an integer seed, or a numpy RandomState")
@icontract.require(lambda width: _positive_int(width), "width must be a positive integer")
@icontract.ensure(
    lambda result, width: _same_vector_shape_and_bounds(result, width),
    "result must be a finite vector in [-1, 1] with the requested width",
)
def bicluster_svd_arpack_init_vector(
    random_state: object,
    width: int,
) -> NDArray[np.float64]:
    from sklearn.utils import check_random_state
    """Build the ARPACK-style initialization vector used for eigsh fallback."""
    rng = check_random_state(random_state)
    return np.asarray(rng.uniform(-1, 1, int(width)), dtype=np.float64)

@register_atom(witness_bicluster_svd_eigsh_kwargs)
@icontract.require(lambda n_svd_vecs: _optional_positive_int(n_svd_vecs), "n_svd_vecs must be None or a positive integer")
@icontract.require(lambda v0: _finite_vector(v0), "v0 must be a finite nonempty vector")
@icontract.ensure(
    lambda result, n_svd_vecs, v0: _eigsh_kwargs_valid(result, n_svd_vecs, v0),
    "result must match the eigsh kwargs used by BaseSpectral._svd NaN recovery",
)
def bicluster_svd_eigsh_kwargs(
    n_svd_vecs: int | None,
    v0: NDArray[np.float64],
) -> dict[str, object]:
    """Resolve the eigsh kwargs used during BaseSpectral._svd NaN recovery."""
    return {
        "ncv": n_svd_vecs,
        "v0": v0,
    }
