"""Shared covariance post-fit API helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from sklearn.utils import check_array

from sciona.ghost.registry import register_atom

from ..atoms import empirical_covariance
from ..state_models import CovarianceState
from .witnesses import (
    witness_covariance_mahalanobis_location_row,
    witness_covariance_mahalanobis_result,
    witness_covariance_precision_matrix,
    witness_covariance_score_test_covariance,
)


Matrix = NDArray[np.float64]
Vector = NDArray[np.float64]


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


def _finite_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _finite_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _covariance_state_valid(state: CovarianceState) -> bool:
    covariance = np.asarray(state.covariance, dtype=np.float64)
    location = np.asarray(state.location, dtype=np.float64)
    precision = None if state.precision is None else np.asarray(state.precision, dtype=np.float64)
    return bool(
        covariance.ndim == 2
        and covariance.shape[0] >= 1
        and covariance.shape[0] == covariance.shape[1]
        and np.all(np.isfinite(covariance))
        and np.allclose(covariance, covariance.T, equal_nan=True)
        and location.shape == (covariance.shape[0],)
        and np.all(np.isfinite(location))
        and (
            (not state.store_precision and precision is None)
            or (
                state.store_precision
                and precision is not None
                and precision.shape == covariance.shape
                and np.all(np.isfinite(precision))
            )
        )
    )


def _matches_location_width(X_test: Matrix, location: Vector) -> bool:
    return np.asarray(X_test, dtype=np.float64).shape[1] == np.asarray(location, dtype=np.float64).shape[0]


def _single_column_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] == 1 and np.all(np.isfinite(array)))


@register_atom(witness_covariance_precision_matrix)
@icontract.require(lambda state: _covariance_state_valid(state), "state must contain a finite covariance, location, and optional stored precision")
@icontract.ensure(lambda result: _finite_square_matrix(result), "precision must be a finite square matrix")
def covariance_precision_matrix(state: CovarianceState) -> Matrix:
    """Resolve sklearn's stored-versus-derived precision matrix for covariance estimators."""
    if state.store_precision:
        return np.asarray(state.precision, dtype=np.float64)
    return np.asarray(linalg.pinvh(np.asarray(state.covariance, dtype=np.float64), check_finite=False), dtype=np.float64)


@register_atom(witness_covariance_score_test_covariance)
@icontract.require(lambda X_test: _finite_matrix(X_test), "X_test must be a finite nonempty 2D matrix")
@icontract.require(lambda location: _finite_vector(location), "location must be a finite nonempty vector")
@icontract.require(lambda X_test, location: _matches_location_width(X_test, location), "location width must match X_test feature count")
@icontract.ensure(lambda result: _finite_square_matrix(result), "test covariance must be a finite square matrix")
def covariance_score_test_covariance(
    X_test: Matrix,
    location: Vector,
) -> Matrix:
    """Compute sklearn's centered test covariance before covariance-estimator scoring."""
    checked_x = np.asarray(check_array(X_test), dtype=np.float64)
    centered = checked_x - np.asarray(location, dtype=np.float64)
    return np.asarray(empirical_covariance(centered, assume_centered=True), dtype=np.float64)


@register_atom(witness_covariance_mahalanobis_location_row)
@icontract.require(lambda location: _finite_vector(location), "location must be a finite nonempty vector")
@icontract.ensure(lambda result: _finite_matrix(result), "location row must be a finite matrix")
@icontract.ensure(lambda result, location: result.shape == (1, np.asarray(location, dtype=np.float64).shape[0]), "location row must add a leading singleton axis")
def covariance_mahalanobis_location_row(location: Vector) -> Matrix:
    """Add sklearn's singleton row axis to a fitted covariance location vector."""
    return np.asarray(np.asarray(location, dtype=np.float64)[np.newaxis, :], dtype=np.float64)


@register_atom(witness_covariance_mahalanobis_result)
@icontract.require(lambda distances: _single_column_matrix(distances), "distances must be a finite matrix with one column")
@icontract.ensure(lambda result: _finite_vector(result), "Mahalanobis output must be a finite vector")
@icontract.ensure(lambda result, distances: result.shape == (np.asarray(distances, dtype=np.float64).shape[0],), "Mahalanobis output must have one entry per row")
def covariance_mahalanobis_result(distances: Matrix) -> Vector:
    """Flatten sklearn's pairwise Mahalanobis distances and square them."""
    values = np.asarray(distances, dtype=np.float64)
    return np.asarray(np.reshape(values, (len(values),)) ** 2, dtype=np.float64)
