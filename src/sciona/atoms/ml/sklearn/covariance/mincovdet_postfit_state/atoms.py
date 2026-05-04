"""MinCovDet post-fit state atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_mincovdet_fit_covariance,
    witness_mincovdet_fit_distances,
    witness_mincovdet_fit_location,
    witness_mincovdet_fit_raw_covariance,
    witness_mincovdet_fit_raw_location,
    witness_mincovdet_fit_raw_support,
    witness_mincovdet_fit_return_self,
    witness_mincovdet_fit_support,
)


def _finite_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


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


def _bool_vector(values: object) -> bool:
    array = np.asarray(values)
    return bool(array.ndim == 1 and array.shape[0] >= 1 and array.dtype == np.bool_)


def _nonnegative_vector(values: object) -> bool:
    return bool(_finite_vector(values) and np.all(np.asarray(values, dtype=np.float64) >= 0.0))


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and value != ""


@register_atom(witness_mincovdet_fit_raw_location)
@icontract.require(lambda raw_location: _finite_vector(raw_location), "raw_location must be a finite nonempty vector")
@icontract.ensure(
    lambda result, raw_location: _finite_vector(result) and np.asarray(result).shape == np.asarray(raw_location).shape,
    "raw_location must preserve the fitted raw-location shape",
)
def mincovdet_fit_raw_location(
    raw_location: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Expose MinCovDet.fit's fitted raw_location_ vector."""
    return np.asarray(raw_location, dtype=np.float64)


@register_atom(witness_mincovdet_fit_raw_covariance)
@icontract.require(lambda raw_covariance: _finite_square_matrix(raw_covariance), "raw_covariance must be a finite nonempty square matrix")
@icontract.ensure(
    lambda result, raw_covariance: _finite_square_matrix(result) and np.asarray(result).shape == np.asarray(raw_covariance).shape,
    "raw_covariance must preserve the fitted raw-covariance shape",
)
def mincovdet_fit_raw_covariance(
    raw_covariance: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Expose MinCovDet.fit's fitted raw_covariance_ matrix."""
    return np.asarray(raw_covariance, dtype=np.float64)


@register_atom(witness_mincovdet_fit_raw_support)
@icontract.require(lambda raw_support: _bool_vector(raw_support), "raw_support must be a nonempty boolean vector")
@icontract.ensure(
    lambda result, raw_support: _bool_vector(result) and np.asarray(result).shape == np.asarray(raw_support).shape,
    "raw_support must preserve the fitted raw-support shape",
)
def mincovdet_fit_raw_support(
    raw_support: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Expose MinCovDet.fit's fitted raw_support_ mask."""
    return np.asarray(raw_support, dtype=np.bool_)


@register_atom(witness_mincovdet_fit_location)
@icontract.require(lambda location: _finite_vector(location), "location must be a finite nonempty vector")
@icontract.ensure(
    lambda result, location: _finite_vector(result) and np.asarray(result).shape == np.asarray(location).shape,
    "location must preserve the fitted location_ shape",
)
def mincovdet_fit_location(
    location: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Expose MinCovDet.fit's fitted location_ vector."""
    return np.asarray(location, dtype=np.float64)


@register_atom(witness_mincovdet_fit_covariance)
@icontract.require(lambda covariance: _finite_square_matrix(covariance), "covariance must be a finite nonempty square matrix")
@icontract.ensure(
    lambda result, covariance: _finite_square_matrix(result) and np.asarray(result).shape == np.asarray(covariance).shape,
    "covariance must preserve the fitted covariance_ shape",
)
def mincovdet_fit_covariance(
    covariance: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Expose MinCovDet.fit's fitted covariance_ matrix."""
    return np.asarray(covariance, dtype=np.float64)


@register_atom(witness_mincovdet_fit_support)
@icontract.require(lambda support: _bool_vector(support), "support must be a nonempty boolean vector")
@icontract.ensure(
    lambda result, support: _bool_vector(result) and np.asarray(result).shape == np.asarray(support).shape,
    "support must preserve the fitted support_ shape",
)
def mincovdet_fit_support(
    support: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Expose MinCovDet.fit's fitted support_ mask."""
    return np.asarray(support, dtype=np.bool_)


@register_atom(witness_mincovdet_fit_distances)
@icontract.require(lambda distances: _nonnegative_vector(distances), "distances must be a finite nonnegative vector")
@icontract.ensure(
    lambda result, distances: _nonnegative_vector(result) and np.asarray(result).shape == np.asarray(distances).shape,
    "distances must preserve the fitted dist_ shape",
)
def mincovdet_fit_distances(
    distances: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Expose MinCovDet.fit's fitted dist_ vector."""
    return np.asarray(distances, dtype=np.float64)


@register_atom(witness_mincovdet_fit_return_self)
@icontract.require(lambda estimator_token: _nonempty_string(estimator_token), "estimator_token must be a nonempty string")
@icontract.ensure(lambda result, estimator_token: result == estimator_token, "fit return value must preserve the estimator token")
def mincovdet_fit_return_self(estimator_token: str) -> str:
    """Model MinCovDet.fit returning self after fitted-state assignment."""
    return estimator_token
