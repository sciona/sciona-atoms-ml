"""Ghost witnesses for EllipticEnvelope post-fit state atoms."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def witness_elliptic_envelope_fit_raw_location(
    raw_location: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Describe the inherited fitted raw robust location vector."""
    return np.asarray(raw_location, dtype=np.float64)


def witness_elliptic_envelope_fit_raw_covariance(
    raw_covariance: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Describe the inherited fitted raw robust covariance matrix."""
    return np.asarray(raw_covariance, dtype=np.float64)


def witness_elliptic_envelope_fit_raw_support(
    raw_support: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Describe the inherited fitted raw support mask."""
    return np.asarray(raw_support, dtype=np.bool_)


def witness_elliptic_envelope_fit_location(
    location: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Describe the inherited fitted final location vector."""
    return np.asarray(location, dtype=np.float64)


def witness_elliptic_envelope_fit_covariance(
    covariance: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Describe the inherited fitted final covariance matrix."""
    return np.asarray(covariance, dtype=np.float64)


def witness_elliptic_envelope_fit_precision(
    precision: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Describe the inherited fitted precision matrix."""
    return np.asarray(precision, dtype=np.float64)


def witness_elliptic_envelope_fit_support(
    support: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Describe the inherited fitted final support mask."""
    return np.asarray(support, dtype=np.bool_)


def witness_elliptic_envelope_fit_distances(
    distances: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Describe the inherited fitted robust squared-distance vector."""
    return np.asarray(distances, dtype=np.float64)


def witness_elliptic_envelope_fit_offset(offset: float) -> float:
    """Describe the fitted EllipticEnvelope offset scalar."""
    return float(offset)


def witness_elliptic_envelope_fit_return_self(estimator_token: str) -> str:
    """Describe the final self-return after EllipticEnvelope fit state assignment."""
    return estimator_token
