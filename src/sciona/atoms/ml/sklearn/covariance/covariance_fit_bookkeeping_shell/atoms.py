"""Shared covariance fit-bookkeeping atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from sklearn.utils import check_array

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_covariance_fit_location,
    witness_covariance_set_covariance_matrix,
    witness_covariance_set_precision_matrix,
    witness_covariance_set_precision_required,
)


Matrix = NDArray[np.float64]
Vector = NDArray[np.float64]


def _finite_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _finite_square_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[0] == array.shape[1] and np.all(np.isfinite(array)))


def _finite_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _bool_scalar(value: object) -> bool:
    return isinstance(value, bool)


def _location_matches_X(result: object, X: object) -> bool:
    return bool(
        _finite_vector(result)
        and _finite_matrix(X)
        and np.asarray(result, dtype=np.float64).shape == (np.asarray(X, dtype=np.float64).shape[1],)
    )


@register_atom(witness_covariance_fit_location)
@icontract.require(lambda X: _finite_matrix(X), "X must be a finite nonempty 2D matrix")
@icontract.require(lambda assume_centered: _bool_scalar(assume_centered), "assume_centered must be boolean")
@icontract.ensure(lambda result, X: _location_matches_X(result, X), "location must be a finite feature-length vector")
def covariance_fit_location(
    X: Matrix,
    *,
    assume_centered: bool,
) -> Vector:
    """Resolve sklearn covariance estimators' fitted location vector during fit."""
    values = np.asarray(X, dtype=np.float64)
    if assume_centered:
        return np.zeros(values.shape[1], dtype=np.float64)
    return np.asarray(values.mean(axis=0), dtype=np.float64)


@register_atom(witness_covariance_set_covariance_matrix)
@icontract.require(lambda covariance: _finite_square_matrix(covariance), "covariance must be a finite nonempty square matrix")
@icontract.ensure(
    lambda result, covariance: _finite_square_matrix(result) and np.array_equal(np.asarray(result, dtype=np.float64), np.asarray(covariance, dtype=np.float64)),
    "result must expose covariance_ unchanged after validation",
)
def covariance_set_covariance_matrix(
    covariance: Matrix,
) -> Matrix:
    """Model sklearn's validated covariance_ assignment inside `_set_covariance`."""
    return np.asarray(check_array(covariance), dtype=np.float64)


@register_atom(witness_covariance_set_precision_required)
@icontract.require(lambda store_precision: _bool_scalar(store_precision), "store_precision must be boolean")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def covariance_set_precision_required(
    store_precision: bool,
) -> bool:
    """Decide whether sklearn stores a fitted precision matrix."""
    return bool(store_precision)


@register_atom(witness_covariance_set_precision_matrix)
@icontract.require(lambda covariance: _finite_square_matrix(covariance), "covariance must be a finite nonempty square matrix")
@icontract.ensure(lambda result: _finite_square_matrix(result), "precision must be a finite nonempty square matrix")
def covariance_set_precision_matrix(
    covariance: Matrix,
) -> Matrix:
    """Compute sklearn's precision_ from covariance_ inside `_set_covariance`."""
    return np.asarray(linalg.pinvh(np.asarray(covariance, dtype=np.float64), check_finite=False), dtype=np.float64)
