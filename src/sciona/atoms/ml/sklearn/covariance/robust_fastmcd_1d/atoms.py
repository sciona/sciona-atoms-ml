"""One-dimensional FastMCD helper atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_fast_mcd_1d_covariance,
    witness_fast_mcd_1d_location,
    witness_fast_mcd_1d_squared_distances,
    witness_fast_mcd_1d_support_mask,
    witness_fast_mcd_support_count,
)


def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _support_fraction_valid(value: float | None) -> bool:
    return bool(
        value is None
        or (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and np.isfinite(float(value))
            and 0.0 < float(value) <= 1.0
        )
    )


def _finite_vector_or_column(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if array.ndim == 1:
        return bool(array.shape[0] >= 2 and np.all(np.isfinite(array)))
    return bool(
        array.ndim == 2
        and array.shape[0] >= 2
        and array.shape[1] == 1
        and np.all(np.isfinite(array))
    )


def _column_values(X: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(X, dtype=np.float64)
    if values.ndim == 1:
        return values.reshape(-1, 1)
    return values


def _support_count_inputs_valid(n_samples: int, n_features: int, support_fraction: float | None) -> bool:
    return bool(_positive_int(n_samples) and _positive_int(n_features) and _support_fraction_valid(support_fraction))


def _support_count_valid(result: int, n_samples: int) -> bool:
    return bool(_positive_int(result) and int(result) <= int(n_samples))


def _location_inputs_valid(X: NDArray[np.float64], n_support: int) -> bool:
    values = _column_values(np.asarray(X, dtype=np.float64))
    return bool(_finite_vector_or_column(values) and _positive_int(n_support) and int(n_support) <= values.shape[0])


def _location_valid(result: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(values.shape == (1,) and np.all(np.isfinite(values)))


def _support_mask_inputs_valid(X: NDArray[np.float64], location: NDArray[np.float64], n_support: int) -> bool:
    values = _column_values(np.asarray(X, dtype=np.float64))
    loc = np.asarray(location, dtype=np.float64)
    return bool(
        _finite_vector_or_column(values)
        and loc.shape == (1,)
        and np.all(np.isfinite(loc))
        and _positive_int(n_support)
        and int(n_support) <= values.shape[0]
    )


def _support_mask_valid(result: NDArray[np.bool_], X: NDArray[np.float64], n_support: int) -> bool:
    values = np.asarray(result, dtype=np.bool_)
    return bool(values.shape == (_column_values(np.asarray(X, dtype=np.float64)).shape[0],) and int(values.sum()) == int(n_support))


def _covariance_inputs_valid(X: NDArray[np.float64], support_mask: NDArray[np.bool_]) -> bool:
    values = _column_values(np.asarray(X, dtype=np.float64))
    mask = np.asarray(support_mask, dtype=np.bool_)
    return bool(values.shape[0] == mask.shape[0] and np.any(mask))


def _covariance_valid(result: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(values.shape == (1, 1) and np.all(np.isfinite(values)) and values[0, 0] >= 0.0)


def _distance_inputs_valid(X: NDArray[np.float64], location: NDArray[np.float64], covariance: NDArray[np.float64]) -> bool:
    values = _column_values(np.asarray(X, dtype=np.float64))
    loc = np.asarray(location, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    return bool(
        _finite_vector_or_column(values)
        and loc.shape == (1,)
        and np.all(np.isfinite(loc))
        and cov.shape == (1, 1)
        and np.all(np.isfinite(cov))
        and cov[0, 0] >= 0.0
    )


def _distance_valid(result: NDArray[np.float64], X: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(values.shape == (_column_values(np.asarray(X, dtype=np.float64)).shape[0],) and np.all(np.isfinite(values)) and np.all(values >= 0.0))


@register_atom(witness_fast_mcd_support_count)
@icontract.require(lambda n_samples, n_features, support_fraction=None: _support_count_inputs_valid(n_samples, n_features, support_fraction), "n_samples and n_features must be positive integers, and support_fraction must be None or in (0, 1]")
@icontract.ensure(lambda result, n_samples: _support_count_valid(result, n_samples), "support count must be positive and no larger than n_samples")
def fast_mcd_support_count(
    n_samples: int,
    n_features: int,
    *,
    support_fraction: float | None = None,
) -> int:
    """Resolve the support size used by sklearn's FastMCD."""
    if support_fraction is None:
        return int(np.ceil(0.5 * (int(n_samples) + int(n_features) + 1)))
    return int(float(support_fraction) * int(n_samples))


@register_atom(witness_fast_mcd_1d_location)
@icontract.require(lambda X, n_support: _location_inputs_valid(X, n_support), "X must be a finite one-dimensional sample vector or column matrix, and n_support must lie between one and n_samples")
@icontract.ensure(lambda result: _location_valid(result), "location must have shape (1,) and be finite")
def fast_mcd_1d_location(
    X: NDArray[np.float64],
    n_support: int,
) -> NDArray[np.float64]:
    """Compute sklearn's one-dimensional FastMCD raw location estimate."""
    values = np.ravel(np.asarray(X, dtype=np.float64))
    support_count = int(n_support)
    n_samples = values.shape[0]
    if support_count < n_samples:
        sorted_values = np.sort(values)
        diff = sorted_values[support_count:] - sorted_values[: (n_samples - support_count)]
        halves_start = np.where(diff == np.min(diff))[0]
        location = 0.5 * (sorted_values[support_count + halves_start] + sorted_values[halves_start]).mean()
        return np.asarray([location], dtype=np.float64)
    return np.asarray([np.mean(values)], dtype=np.float64)


@register_atom(witness_fast_mcd_1d_support_mask)
@icontract.require(lambda X, location, n_support: _support_mask_inputs_valid(X, location, n_support), "X, location, and n_support must describe a valid one-dimensional FastMCD support selection")
@icontract.ensure(lambda result, X, n_support: _support_mask_valid(result, X, n_support), "support mask must match the sample count and contain exactly n_support true values")
def fast_mcd_1d_support_mask(
    X: NDArray[np.float64],
    location: NDArray[np.float64],
    n_support: int,
) -> NDArray[np.bool_]:
    """Select sklearn's one-dimensional FastMCD support from absolute centered distances."""
    values = _column_values(np.asarray(X, dtype=np.float64))
    centered = values - np.asarray(location, dtype=np.float64)
    support = np.zeros(values.shape[0], dtype=np.bool_)
    support[np.argsort(np.abs(centered), axis=0)[: int(n_support)].ravel()] = True
    return support


@register_atom(witness_fast_mcd_1d_covariance)
@icontract.require(lambda X, support_mask: _covariance_inputs_valid(X, support_mask), "X and support_mask must be compatible and support_mask must select at least one sample")
@icontract.ensure(lambda result: _covariance_valid(result), "covariance must be a finite nonnegative 1x1 matrix")
def fast_mcd_1d_covariance(
    X: NDArray[np.float64],
    support_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Compute sklearn's one-dimensional FastMCD raw covariance estimate."""
    values = _column_values(np.asarray(X, dtype=np.float64))
    mask = np.asarray(support_mask, dtype=np.bool_)
    return np.asarray([[np.var(values[mask])]], dtype=np.float64)


@register_atom(witness_fast_mcd_1d_squared_distances)
@icontract.require(lambda X, location, covariance: _distance_inputs_valid(X, location, covariance), "X, location, and covariance must describe a valid one-dimensional FastMCD distance calculation")
@icontract.ensure(lambda result, X: _distance_valid(result, X), "squared distances must be finite, nonnegative, and have one entry per sample")
def fast_mcd_1d_squared_distances(
    X: NDArray[np.float64],
    location: NDArray[np.float64],
    covariance: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute sklearn's one-dimensional FastMCD squared Mahalanobis distances."""
    values = _column_values(np.asarray(X, dtype=np.float64))
    centered = values - np.asarray(location, dtype=np.float64)
    precision = linalg.pinvh(np.asarray(covariance, dtype=np.float64))
    return np.asarray((np.dot(centered, precision) * centered).sum(axis=1), dtype=np.float64)
