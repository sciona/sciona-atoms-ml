"""MinCovDet fit-prelude atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_mincovdet_fit_assume_centered_branch,
    witness_mincovdet_fit_random_state,
    witness_mincovdet_fit_shape,
    witness_mincovdet_fit_validated_data,
)


Matrix = NDArray[np.float64]


def _array_like_2d(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 2 and array.shape[1] >= 1)


def _finite_matrix(values: object) -> bool:
    if not _array_like_2d(values):
        return False
    array = np.asarray(values, dtype=np.float64)
    return bool(np.all(np.isfinite(array)))


def _same_shape_and_values(result: object, expected: object) -> bool:
    lhs = np.asarray(result, dtype=np.float64)
    rhs = np.asarray(expected, dtype=np.float64)
    return bool(lhs.shape == rhs.shape and np.array_equal(lhs, rhs))


def _valid_random_state_like(value: object) -> bool:
    return bool(
        value is None
        or isinstance(value, (int, np.integer, np.random.RandomState))
    )


def _random_state_result_valid(result: object) -> bool:
    return isinstance(result, np.random.RandomState)


def _shape_result_valid(result: object, X: object) -> bool:
    if not isinstance(result, tuple) or len(result) != 2:
        return False
    if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 1 for v in result):
        return False
    values = np.asarray(X, dtype=np.float64)
    return result == (int(values.shape[0]), int(values.shape[1]))


def _bool_value(value: object) -> bool:
    return isinstance(value, bool)


@register_atom(witness_mincovdet_fit_validated_data)
@icontract.require(lambda X: _array_like_2d(X), "X must be a 2D float-like matrix with at least two rows and one column")
@icontract.ensure(
    lambda result: _finite_matrix(result),
    "validated data must be a finite float64 matrix",
)
def mincovdet_fit_validated_data(
    X: object,
) -> Matrix:
    """Validate MinCovDet.fit input with sklearn's dense array checks."""
    return np.asarray(check_array(X, ensure_min_samples=2, dtype=np.float64), dtype=np.float64)


@register_atom(witness_mincovdet_fit_random_state)
@icontract.require(lambda random_state: _valid_random_state_like(random_state), "random_state must be None, an integer seed, or a numpy RandomState")
@icontract.ensure(lambda result: _random_state_result_valid(result), "result must be a numpy RandomState")
def mincovdet_fit_random_state(
    random_state: object,
) -> np.random.RandomState:
    """Normalize MinCovDet.fit's random_state with sklearn's check_random_state."""
    return check_random_state(random_state)


@register_atom(witness_mincovdet_fit_shape)
@icontract.require(lambda X: _finite_matrix(X), "X must be a finite 2D matrix with at least two rows and one column")
@icontract.ensure(lambda result, X: _shape_result_valid(result, X), "result must be the `(n_samples, n_features)` tuple for X")
def mincovdet_fit_shape(
    X: Matrix,
) -> tuple[int, int]:
    """Unpack the `(n_samples, n_features)` pair used by MinCovDet.fit."""
    values = np.asarray(X, dtype=np.float64)
    return int(values.shape[0]), int(values.shape[1])


@register_atom(witness_mincovdet_fit_assume_centered_branch)
@icontract.require(lambda assume_centered: _bool_value(assume_centered), "assume_centered must be boolean")
@icontract.ensure(lambda result: _bool_value(result), "branch predicate must be boolean")
def mincovdet_fit_assume_centered_branch(
    assume_centered: bool,
) -> bool:
    """Return whether MinCovDet.fit enters the assume-centered raw-estimate branch."""
    return bool(assume_centered)
