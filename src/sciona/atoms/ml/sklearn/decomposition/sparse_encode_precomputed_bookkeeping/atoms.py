"""Sparse-encode precomputed bookkeeping atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_sparse_encode_lasso_alpha,
    witness_sparse_encode_omp_norms_squared,
    witness_sparse_encode_precomputed_output,
    witness_sparse_encode_writable_init,
)

def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)

def _finite_nonnegative_scalar(value: object) -> bool:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(scalar) and scalar >= 0.0)

def _matrix_valid(value: object) -> bool:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 2
        and np.issubdtype(array.dtype, np.number)
        and np.all(np.isfinite(array))
    )

def _writable_matrix_valid(value: object, template: object) -> bool:
    if not _matrix_valid(value) or not _matrix_valid(template):
        return False
    result = np.asarray(value)
    source = np.asarray(template)
    return bool(
        result.shape == source.shape
        and np.array_equal(result, source)
        and result.flags["WRITEABLE"]
    )

def _norm_vector_valid(result: object, X: object) -> bool:
    if not _matrix_valid(X):
        return False
    values = np.asarray(result, dtype=np.float64)
    X_values = np.asarray(X, dtype=np.float64)
    return bool(
        values.shape == (X_values.shape[0],)
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
    )

def _reshaped_output_valid(result: object, new_code: object, n_samples: int, n_components: int) -> bool:
    try:
        values = np.asarray(result, dtype=np.float64)
        flat_input = np.asarray(new_code, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        values.shape == (int(n_samples), int(n_components))
        and np.all(np.isfinite(values))
        and flat_input.size == int(n_samples) * int(n_components)
        and np.array_equal(values, flat_input.reshape(int(n_samples), int(n_components)))
    )

@register_atom(witness_sparse_encode_lasso_alpha)
@icontract.require(lambda regularization: _finite_nonnegative_scalar(regularization), "regularization must be a finite nonnegative scalar")
@icontract.require(lambda n_features: _positive_int(n_features), "n_features must be a positive integer")
@icontract.ensure(lambda result: isinstance(result, float) and np.isfinite(result) and result >= 0.0, "result must be a finite nonnegative float")
def sparse_encode_lasso_alpha(
    regularization: float,
    n_features: int,
) -> float:
    """Scale sklearn's lasso regularization by the feature count."""
    return float(regularization) / int(n_features)

@register_atom(witness_sparse_encode_writable_init)
@icontract.require(lambda init: _matrix_valid(init), "init must be a finite numeric rank-2 array")
@icontract.ensure(lambda result, init: _writable_matrix_valid(result, init), "result must preserve init values and be writable")
def sparse_encode_writable_init(
    init: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Return a writable lasso_cd warm-start matrix like sklearn's guard."""
    if not init.flags["WRITEABLE"]:
        return np.array(init)
    return init

@register_atom(witness_sparse_encode_omp_norms_squared)
@icontract.require(lambda X: _matrix_valid(X), "X must be a finite numeric rank-2 array")
@icontract.ensure(lambda result, X: _norm_vector_valid(result, X), "result must be one squared norm per sample")
def sparse_encode_omp_norms_squared(
    X: NDArray[np.floating],
) -> NDArray[np.float64]:
    from sklearn.utils.extmath import row_norms
    """Compute sklearn's squared sample norms for the OMP branch."""
    return np.asarray(row_norms(X, squared=True), dtype=np.float64)

@register_atom(witness_sparse_encode_precomputed_output)
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be a positive integer")
@icontract.require(lambda n_components: _positive_int(n_components), "n_components must be a positive integer")
@icontract.require(
    lambda new_code, n_samples, n_components: np.asarray(new_code).size == int(n_samples) * int(n_components),
    "new_code must contain exactly n_samples * n_components values",
)
@icontract.require(lambda new_code: np.all(np.isfinite(np.asarray(new_code, dtype=np.float64))), "new_code must be finite")
@icontract.ensure(
    lambda result, new_code, n_samples, n_components: _reshaped_output_valid(result, new_code, n_samples, n_components),
    "result must reshape new_code into the sparse_encode output matrix",
)
def sparse_encode_precomputed_output(
    new_code: NDArray[np.floating],
    n_samples: int,
    n_components: int,
) -> NDArray[np.float64]:
    """Apply sklearn's final sparse_encode reshape to solver output."""
    return np.asarray(new_code, dtype=np.float64).reshape(int(n_samples), int(n_components))
