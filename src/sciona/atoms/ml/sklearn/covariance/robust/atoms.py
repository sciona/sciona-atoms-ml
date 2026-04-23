"""Deterministic robust-covariance helper atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2

from sciona.atoms.ml.sklearn.covariance import empirical_covariance
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_mcd_consistency_factor,
    witness_mcd_correct_covariance,
    witness_mcd_reweighted_location_covariance,
    witness_mcd_reweight_support_mask,
    witness_mcd_squared_mahalanobis,
)

MCDCorrection = tuple[NDArray[np.float64], NDArray[np.float64]]
MCDReweighting = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]


def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _unit_open(value: float) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)) and 0.0 < float(value) < 1.0)


def _finite_vector(values: NDArray[np.float64]) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _finite_matrix(values: NDArray[np.float64]) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _finite_square_matrix(values: NDArray[np.float64]) -> bool:
    return bool(_finite_matrix(values) and np.asarray(values).shape[0] == np.asarray(values).shape[1])


def _support_mask_valid(mask: NDArray[np.bool_], n_samples: int) -> bool:
    try:
        values = np.asarray(mask, dtype=np.bool_)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 1 and values.shape == (n_samples,) and np.any(values))


def _correct_inputs_valid(raw_covariance: NDArray[np.float64], dist: NDArray[np.float64], n_support: int) -> bool:
    return bool(
        _finite_square_matrix(raw_covariance)
        and _finite_vector(dist)
        and _positive_int(n_support)
        and n_support <= np.asarray(dist).shape[0]
        and np.all(np.asarray(dist, dtype=np.float64) >= 0.0)
    )


def _correction_valid(result: MCDCorrection, raw_covariance: NDArray[np.float64], dist: NDArray[np.float64]) -> bool:
    if not (isinstance(result, tuple) and len(result) == 2):
        return False
    covariance, corrected_dist = result
    return bool(
        np.asarray(covariance).shape == np.asarray(raw_covariance).shape
        and np.asarray(corrected_dist).shape == np.asarray(dist).shape
        and np.all(np.isfinite(covariance))
        and np.all(np.isfinite(corrected_dist))
        and np.all(np.asarray(corrected_dist, dtype=np.float64) >= 0.0)
        and np.allclose(covariance, np.asarray(covariance).T)
    )


def _reweight_inputs_valid(data: NDArray[np.float64], support_mask: NDArray[np.bool_]) -> bool:
    return bool(_finite_matrix(data) and _support_mask_valid(support_mask, np.asarray(data).shape[0]))


def _reweight_result_valid(result: MCDReweighting, data: NDArray[np.float64]) -> bool:
    if not (isinstance(result, tuple) and len(result) == 3):
        return False
    location, covariance, support = result
    n_samples, n_features = np.asarray(data).shape
    return bool(
        np.asarray(location).shape == (n_features,)
        and np.asarray(covariance).shape == (n_features, n_features)
        and np.asarray(support).shape == (n_samples,)
        and np.asarray(support).dtype == np.bool_
        and np.any(support)
        and np.all(np.isfinite(location))
        and np.all(np.isfinite(covariance))
        and np.allclose(covariance, np.asarray(covariance).T)
    )


def _mahalanobis_inputs_valid(X: NDArray[np.float64], location: NDArray[np.float64], precision: NDArray[np.float64]) -> bool:
    if not (_finite_matrix(X) and _finite_vector(location) and _finite_square_matrix(precision)):
        return False
    x_values = np.asarray(X)
    return bool(location.shape == (x_values.shape[1],) and precision.shape == (x_values.shape[1], x_values.shape[1]))


def _distance_valid(result: NDArray[np.float64], X: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(values.shape == (np.asarray(X).shape[0],) and np.all(np.isfinite(values)) and np.all(values >= 0.0))


@register_atom(witness_mcd_consistency_factor)
@icontract.require(lambda n_features: _positive_int(n_features), "feature count must be a positive integer")
@icontract.require(lambda alpha: _unit_open(alpha), "alpha must lie strictly between zero and one")
@icontract.ensure(lambda result: np.isfinite(result) and result > 0.0, "consistency factor must be positive and finite")
def mcd_consistency_factor(n_features: int, alpha: float) -> float:
    """Compute the MCD normal-consistency scaling factor used by sklearn."""
    q_alpha = chi2.ppf(float(alpha), df=n_features)
    return float(float(alpha) / chi2.cdf(q_alpha, n_features + 2))


@register_atom(witness_mcd_correct_covariance)
@icontract.require(lambda raw_covariance, dist, n_support: _correct_inputs_valid(raw_covariance, dist, n_support), "raw covariance, distances, and support count must be compatible")
@icontract.ensure(lambda result, raw_covariance, dist: _correction_valid(result, raw_covariance, dist), "corrected covariance and distances must preserve input shapes")
def mcd_correct_covariance(
    raw_covariance: NDArray[np.float64],
    dist: NDArray[np.float64],
    n_support: int,
) -> MCDCorrection:
    """Apply sklearn's MCD consistency correction to covariance and distances."""
    n_samples = np.asarray(dist).shape[0]
    factor = mcd_consistency_factor(np.asarray(raw_covariance).shape[0], n_support / n_samples)
    covariance_corrected = np.asarray(raw_covariance, dtype=np.float64) * factor
    corrected_dist = np.asarray(dist, dtype=np.float64) / factor
    return covariance_corrected, corrected_dist


@register_atom(witness_mcd_reweight_support_mask)
@icontract.require(lambda dist: _finite_vector(dist) and np.all(np.asarray(dist, dtype=np.float64) >= 0.0), "distances must be finite and nonnegative")
@icontract.require(lambda n_features: _positive_int(n_features), "feature count must be a positive integer")
@icontract.require(lambda quantile_threshold: _unit_open(quantile_threshold), "quantile threshold must lie strictly between zero and one")
@icontract.ensure(lambda result, dist: np.asarray(result).shape == np.asarray(dist).shape and np.asarray(result).dtype == np.bool_, "support mask must match distance shape")
def mcd_reweight_support_mask(
    dist: NDArray[np.float64],
    n_features: int,
    *,
    quantile_threshold: float = 0.025,
) -> NDArray[np.bool_]:
    """Select observations retained by sklearn's MCD chi-square reweight rule."""
    threshold = chi2(n_features).isf(float(quantile_threshold))
    return np.asarray(np.asarray(dist, dtype=np.float64) < threshold, dtype=np.bool_)


@register_atom(witness_mcd_reweighted_location_covariance)
@icontract.require(lambda data, support_mask: _reweight_inputs_valid(data, support_mask), "data and support mask must be compatible and select at least one row")
@icontract.ensure(lambda result, data: _reweight_result_valid(result, data), "reweighted location, covariance, and support must have expected shapes")
def mcd_reweighted_location_covariance(
    data: NDArray[np.float64],
    support_mask: NDArray[np.bool_],
    *,
    assume_centered: bool = False,
) -> MCDReweighting:
    """Compute sklearn's MCD reweighted location, covariance, and support mask."""
    values = np.asarray(data, dtype=np.float64)
    mask = np.asarray(support_mask, dtype=np.bool_)
    if assume_centered:
        location = np.zeros(values.shape[1], dtype=np.float64)
    else:
        location = values[mask].mean(axis=0)
    covariance = empirical_covariance(values[mask], assume_centered=assume_centered)
    support = np.zeros(values.shape[0], dtype=np.bool_)
    support[mask] = True
    return np.asarray(location, dtype=np.float64), np.asarray(covariance, dtype=np.float64), support


@register_atom(witness_mcd_squared_mahalanobis)
@icontract.require(lambda X, location, precision: _mahalanobis_inputs_valid(X, location, precision), "observations, location, and precision must have compatible feature dimensions")
@icontract.ensure(lambda result, X: _distance_valid(result, X), "squared Mahalanobis distances must be nonnegative finite values")
def mcd_squared_mahalanobis(
    X: NDArray[np.float64],
    location: NDArray[np.float64],
    precision: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute squared Mahalanobis distances from a robust location and precision."""
    centered = np.asarray(X, dtype=np.float64) - np.asarray(location, dtype=np.float64)
    return np.asarray(np.sum(np.dot(centered, np.asarray(precision, dtype=np.float64)) * centered, axis=1), dtype=np.float64)
