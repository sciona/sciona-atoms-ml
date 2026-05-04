"""Permutation-importance preflight-shell atoms adapted from scikit-learn."""

from __future__ import annotations

import numbers

import icontract
import numpy as np
from numpy.typing import NDArray
from sklearn.utils import check_array

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_permutation_importance_checked_array,
    witness_permutation_importance_max_samples_guard_required,
    witness_permutation_importance_use_dataframe_passthrough,
)


Matrix = NDArray[np.float64]


def _bool_scalar(value: object) -> bool:
    return isinstance(value, bool)


def _array_like_2d(values: object) -> bool:
    try:
        array = np.asarray(values)
    except Exception:
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1)


def _finite_or_nan_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and not np.any(np.isinf(array)))


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _supported_max_samples(value: object) -> bool:
    return bool(
        isinstance(value, numbers.Integral) and not isinstance(value, bool) and int(value) >= 1
    ) or bool(
        isinstance(value, float) and np.isfinite(value) and 0.0 < float(value) <= 1.0
    )


@register_atom(witness_permutation_importance_use_dataframe_passthrough)
@icontract.require(lambda has_iloc: _bool_scalar(has_iloc), "has_iloc must be boolean")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def permutation_importance_use_dataframe_passthrough(
    has_iloc: bool,
) -> bool:
    """Decide whether permutation_importance skips check_array for dataframe-like input."""
    return bool(has_iloc)


@register_atom(witness_permutation_importance_checked_array)
@icontract.require(lambda X: _array_like_2d(X), "X must be array-like 2D")
@icontract.ensure(lambda result: _finite_or_nan_matrix(result), "checked array must be a finite-or-NaN 2D matrix")
def permutation_importance_checked_array(
    X: object,
) -> Matrix:
    """Apply sklearn's dense check_array branch for non-dataframe permutation input."""
    return np.asarray(check_array(X, ensure_all_finite="allow-nan", dtype=None), dtype=np.float64)


@register_atom(witness_permutation_importance_max_samples_guard_required)
@icontract.require(lambda max_samples: _supported_max_samples(max_samples), "max_samples must be an integer or a float in (0, 1]")
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be a positive integer")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def permutation_importance_max_samples_guard_required(
    max_samples: int | float,
    n_samples: int,
) -> bool:
    """Decide whether permutation_importance raises on oversized integer max_samples."""
    return bool(
        isinstance(max_samples, numbers.Integral)
        and not isinstance(max_samples, bool)
        and int(max_samples) > int(n_samples)
    )
