"""EllipticEnvelope post-fit state atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_elliptic_envelope_fit_covariance,
    witness_elliptic_envelope_fit_distances,
    witness_elliptic_envelope_fit_location,
    witness_elliptic_envelope_fit_offset,
    witness_elliptic_envelope_fit_precision,
    witness_elliptic_envelope_fit_raw_covariance,
    witness_elliptic_envelope_fit_raw_location,
    witness_elliptic_envelope_fit_raw_support,
    witness_elliptic_envelope_fit_return_self,
    witness_elliptic_envelope_fit_support,
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


def _finite_float(value: object) -> bool:
    return bool(np.isscalar(value) and np.isfinite(float(value)))


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and value != ""


@register_atom(witness_elliptic_envelope_fit_raw_location)
@icontract.require(lambda raw_location: _finite_vector(raw_location), "raw_location must be a finite nonempty vector")
@icontract.ensure(
    lambda result, raw_location: _finite_vector(result) and np.asarray(result).shape == np.asarray(raw_location).shape,
    "raw_location must preserve the fitted raw-location shape",
)
def elliptic_envelope_fit_raw_location(
    raw_location: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Expose EllipticEnvelope.fit's inherited raw_location_ vector."""
    return np.asarray(raw_location, dtype=np.float64)


@register_atom(witness_elliptic_envelope_fit_raw_covariance)
@icontract.require(lambda raw_covariance: _finite_square_matrix(raw_covariance), "raw_covariance must be a finite nonempty square matrix")
@icontract.ensure(
    lambda result, raw_covariance: _finite_square_matrix(result) and np.asarray(result).shape == np.asarray(raw_covariance).shape,
    "raw_covariance must preserve the fitted raw-covariance shape",
)
def elliptic_envelope_fit_raw_covariance(
    raw_covariance: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Expose EllipticEnvelope.fit's inherited raw_covariance_ matrix."""
    return np.asarray(raw_covariance, dtype=np.float64)


@register_atom(witness_elliptic_envelope_fit_raw_support)
@icontract.require(lambda raw_support: _bool_vector(raw_support), "raw_support must be a nonempty boolean vector")
@icontract.ensure(
    lambda result, raw_support: _bool_vector(result) and np.asarray(result).shape == np.asarray(raw_support).shape,
    "raw_support must preserve the fitted raw-support shape",
)
def elliptic_envelope_fit_raw_support(
    raw_support: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Expose EllipticEnvelope.fit's inherited raw_support_ mask."""
    return np.asarray(raw_support, dtype=np.bool_)


@register_atom(witness_elliptic_envelope_fit_location)
@icontract.require(lambda location: _finite_vector(location), "location must be a finite nonempty vector")
@icontract.ensure(
    lambda result, location: _finite_vector(result) and np.asarray(result).shape == np.asarray(location).shape,
    "location must preserve the fitted location_ shape",
)
def elliptic_envelope_fit_location(
    location: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Expose EllipticEnvelope.fit's inherited location_ vector."""
    return np.asarray(location, dtype=np.float64)


@register_atom(witness_elliptic_envelope_fit_covariance)
@icontract.require(lambda covariance: _finite_square_matrix(covariance), "covariance must be a finite nonempty square matrix")
@icontract.ensure(
    lambda result, covariance: _finite_square_matrix(result) and np.asarray(result).shape == np.asarray(covariance).shape,
    "covariance must preserve the fitted covariance_ shape",
)
def elliptic_envelope_fit_covariance(
    covariance: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Expose EllipticEnvelope.fit's inherited covariance_ matrix."""
    return np.asarray(covariance, dtype=np.float64)


@register_atom(witness_elliptic_envelope_fit_precision)
@icontract.require(lambda precision: _finite_square_matrix(precision), "precision must be a finite nonempty square matrix")
@icontract.ensure(
    lambda result, precision: _finite_square_matrix(result) and np.asarray(result).shape == np.asarray(precision).shape,
    "precision must preserve the fitted precision_ shape",
)
def elliptic_envelope_fit_precision(
    precision: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Expose EllipticEnvelope.fit's inherited precision_ matrix."""
    return np.asarray(precision, dtype=np.float64)


@register_atom(witness_elliptic_envelope_fit_support)
@icontract.require(lambda support: _bool_vector(support), "support must be a nonempty boolean vector")
@icontract.ensure(
    lambda result, support: _bool_vector(result) and np.asarray(result).shape == np.asarray(support).shape,
    "support must preserve the fitted support_ shape",
)
def elliptic_envelope_fit_support(
    support: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Expose EllipticEnvelope.fit's inherited support_ mask."""
    return np.asarray(support, dtype=np.bool_)


@register_atom(witness_elliptic_envelope_fit_distances)
@icontract.require(lambda distances: _nonnegative_vector(distances), "distances must be a finite nonnegative vector")
@icontract.ensure(
    lambda result, distances: _nonnegative_vector(result) and np.asarray(result).shape == np.asarray(distances).shape,
    "distances must preserve the fitted dist_ shape",
)
def elliptic_envelope_fit_distances(
    distances: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Expose EllipticEnvelope.fit's inherited dist_ vector."""
    return np.asarray(distances, dtype=np.float64)


@register_atom(witness_elliptic_envelope_fit_offset)
@icontract.require(lambda offset: _finite_float(offset), "offset must be finite")
@icontract.ensure(lambda result: _finite_float(result), "offset must remain finite")
def elliptic_envelope_fit_offset(offset: float) -> float:
    """Expose EllipticEnvelope.fit's fitted offset_ scalar."""
    return float(offset)


@register_atom(witness_elliptic_envelope_fit_return_self)
@icontract.require(lambda estimator_token: _nonempty_string(estimator_token), "estimator_token must be a nonempty string")
@icontract.ensure(lambda result, estimator_token: result == estimator_token, "fit return value must preserve the estimator token")
def elliptic_envelope_fit_return_self(estimator_token: str) -> str:
    """Model EllipticEnvelope.fit returning self after fitted-state assignment."""
    return estimator_token
