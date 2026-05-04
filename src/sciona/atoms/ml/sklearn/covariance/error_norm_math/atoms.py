"""Covariance error-norm helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_covariance_error_matrix,
    witness_covariance_error_result,
    witness_covariance_error_scaled_squared_norm,
    witness_covariance_error_squared_norm,
)


Matrix = NDArray[np.float64]


def _finite_square_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 2
        and array.shape[0] >= 1
        and array.shape[0] == array.shape[1]
        and np.all(np.isfinite(array))
    )


def _same_shape(a: object, b: object) -> bool:
    return np.asarray(a, dtype=np.float64).shape == np.asarray(b, dtype=np.float64).shape


def _same_shape_and_values(result: object, expected: object) -> bool:
    lhs = np.asarray(result, dtype=np.float64)
    rhs = np.asarray(expected, dtype=np.float64)
    return bool(lhs.shape == rhs.shape and np.array_equal(lhs, rhs))


def _norm_valid(value: object) -> bool:
    return isinstance(value, str) and value != ""


def _supported_norm(value: str) -> bool:
    return value in {"frobenius", "spectral"}


def _finite_nonnegative_scalar(value: object) -> bool:
    return bool(
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool)
        and np.isfinite(float(value))
        and float(value) >= 0.0
    )


def _bool_value(value: object) -> bool:
    return isinstance(value, bool)


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


@register_atom(witness_covariance_error_matrix)
@icontract.require(lambda comp_cov: _finite_square_matrix(comp_cov), "comp_cov must be a finite nonempty square matrix")
@icontract.require(lambda covariance: _finite_square_matrix(covariance), "covariance must be a finite nonempty square matrix")
@icontract.require(lambda comp_cov, covariance: _same_shape(comp_cov, covariance), "covariance must match comp_cov shape")
@icontract.ensure(
    lambda result, comp_cov, covariance: _same_shape_and_values(
        result, np.asarray(comp_cov, dtype=np.float64) - np.asarray(covariance, dtype=np.float64)
    ),
    "result must equal comp_cov minus covariance",
)
def covariance_error_matrix(
    comp_cov: Matrix,
    covariance: Matrix,
) -> Matrix:
    """Compute the covariance-difference matrix used by sklearn error_norm."""
    return np.asarray(np.asarray(comp_cov, dtype=np.float64) - np.asarray(covariance, dtype=np.float64), dtype=np.float64)


@register_atom(witness_covariance_error_squared_norm)
@icontract.require(lambda error: _finite_square_matrix(error), "error must be a finite nonempty square matrix")
@icontract.require(lambda norm: _norm_valid(norm), "norm must be a nonempty string")
@icontract.ensure(lambda result: _finite_nonnegative_scalar(result), "squared norm must be a finite nonnegative scalar")
def covariance_error_squared_norm(
    error: Matrix,
    norm: str = "frobenius",
) -> float:
    """Compute sklearn's squared covariance-error norm before optional scaling."""
    values = np.asarray(error, dtype=np.float64)
    if norm == "frobenius":
        return float(np.sum(values**2))
    if norm == "spectral":
        return float(np.amax(linalg.svdvals(np.dot(values.T, values))))
    raise NotImplementedError("Only spectral and frobenius norms are implemented")


@register_atom(witness_covariance_error_scaled_squared_norm)
@icontract.require(lambda squared_norm: _finite_nonnegative_scalar(squared_norm), "squared_norm must be a finite nonnegative scalar")
@icontract.require(lambda scaling: _bool_value(scaling), "scaling must be boolean")
@icontract.require(lambda n_features: _positive_int(n_features), "n_features must be a positive integer")
@icontract.ensure(lambda result: _finite_nonnegative_scalar(result), "scaled squared norm must be a finite nonnegative scalar")
def covariance_error_scaled_squared_norm(
    squared_norm: float,
    scaling: bool = True,
    n_features: int = 1,
) -> float:
    """Apply sklearn's optional feature-count scaling to a squared error norm."""
    if scaling:
        return float(float(squared_norm) / int(n_features))
    return float(squared_norm)


@register_atom(witness_covariance_error_result)
@icontract.require(lambda squared_norm: _finite_nonnegative_scalar(squared_norm), "squared_norm must be a finite nonnegative scalar")
@icontract.require(lambda squared: _bool_value(squared), "squared must be boolean")
@icontract.ensure(lambda result: _finite_nonnegative_scalar(result), "result must be a finite nonnegative scalar")
def covariance_error_result(
    squared_norm: float,
    squared: bool = True,
) -> float:
    """Return sklearn's final error_norm scalar with optional square root."""
    if squared:
        return float(squared_norm)
    return float(np.sqrt(float(squared_norm)))
