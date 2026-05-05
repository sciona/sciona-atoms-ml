"""Sklearn coordinate-descent alpha-grid math atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_alpha_grid_alpha_max,
    witness_cd_alpha_grid_sample_count,
    witness_cd_alpha_grid_use_resolution_fallback,
    witness_cd_alpha_grid_values,
    witness_cd_alpha_grid_xyw_matrix,
)


def _finite_array(values: object, ndim: int | None = None) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if ndim is not None and array.ndim != ndim:
        return False
    return bool(array.size >= 1 and np.all(np.isfinite(array)))


def _positive_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and int(value) >= 1


def _positive_float(value: object) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(numeric) and numeric > 0.0)


@register_atom(witness_cd_alpha_grid_xyw_matrix)
@icontract.require(
    lambda Xyw: _finite_array(Xyw) and np.asarray(Xyw, dtype=np.float64).ndim in {1, 2},
    "Xyw must be a finite one- or two-dimensional numeric array",
)
@icontract.ensure(
    lambda result, Xyw: _finite_array(result, ndim=2)
    and np.asarray(result, dtype=np.float64).shape[0] == np.asarray(Xyw, dtype=np.float64).shape[0]
    and np.asarray(result, dtype=np.float64).shape[1] >= 1,
    "Xyw must be normalized to a two-dimensional finite matrix",
)
def cd_alpha_grid_xyw_matrix(
    Xyw: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Return the two-dimensional Xyw matrix used by _alpha_grid."""
    matrix = np.asarray(Xyw, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, np.newaxis]
    return np.asarray(matrix, dtype=np.float64)


@register_atom(witness_cd_alpha_grid_sample_count)
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be a positive integer")
@icontract.require(
    lambda sample_weight_sum=None: sample_weight_sum is None or _positive_float(sample_weight_sum),
    "sample_weight_sum must be None or a positive finite float",
)
@icontract.ensure(
    lambda result, n_samples, sample_weight_sum=None: _positive_float(result)
    and float(result)
    == (float(sample_weight_sum) if sample_weight_sum is not None else float(n_samples)),
    "sample_count must be sample_weight_sum when provided, else n_samples",
)
def cd_alpha_grid_sample_count(
    n_samples: int,
    sample_weight_sum: float | None = None,
) -> float:
    """Return the effective sample count used by _alpha_grid."""
    if sample_weight_sum is not None:
        return float(sample_weight_sum)
    return float(n_samples)


@register_atom(witness_cd_alpha_grid_alpha_max)
@icontract.require(lambda Xyw: _finite_array(Xyw, ndim=2), "Xyw must be a finite two-dimensional matrix")
@icontract.require(lambda sample_count: _positive_float(sample_count), "sample_count must be a positive finite float")
@icontract.require(lambda l1_ratio: _positive_float(l1_ratio), "l1_ratio must be a positive finite float")
@icontract.ensure(
    lambda result: _positive_float(result) or float(result) == 0.0,
    "alpha_max must be nonnegative and finite",
)
def cd_alpha_grid_alpha_max(
    Xyw: NDArray[np.float64],
    sample_count: float,
    l1_ratio: float,
) -> float:
    """Return the alpha_max value used by _alpha_grid."""
    matrix = np.asarray(Xyw, dtype=np.float64)
    return float(np.sqrt(np.sum(matrix**2, axis=1)).max() / (float(sample_count) * float(l1_ratio)))


@register_atom(witness_cd_alpha_grid_use_resolution_fallback)
@icontract.require(
    lambda alpha_max: np.isfinite(float(alpha_max)) and float(alpha_max) >= 0.0,
    "alpha_max must be a finite nonnegative float",
)
@icontract.ensure(
    lambda result, alpha_max: isinstance(result, bool)
    and result == (float(alpha_max) <= np.finfo(np.float64).resolution),
    "fallback predicate must match the float64 resolution cutoff",
)
def cd_alpha_grid_use_resolution_fallback(alpha_max: float) -> bool:
    """Return whether _alpha_grid should emit the float64-resolution fallback."""
    return bool(float(alpha_max) <= np.finfo(np.float64).resolution)


@register_atom(witness_cd_alpha_grid_values)
@icontract.require(
    lambda alpha_max: np.isfinite(float(alpha_max)) and float(alpha_max) >= 0.0,
    "alpha_max must be a finite nonnegative float",
)
@icontract.require(lambda eps: _positive_float(eps), "eps must be a positive finite float")
@icontract.require(lambda n_alphas: _positive_int(n_alphas), "n_alphas must be a positive integer")
@icontract.ensure(
    lambda result, alpha_max, n_alphas: _finite_array(result, ndim=1)
    and np.asarray(result, dtype=np.float64).shape == (int(n_alphas),)
    and np.all(np.asarray(result, dtype=np.float64) > 0.0),
    "alpha grid must be a positive finite one-dimensional vector of length n_alphas",
)
def cd_alpha_grid_values(
    alpha_max: float,
    eps: float,
    n_alphas: int,
) -> NDArray[np.float64]:
    """Return the final alpha grid produced by _alpha_grid once alpha_max is known."""
    alpha_max = float(alpha_max)
    eps = float(eps)
    n_alphas = int(n_alphas)
    resolution = np.finfo(np.float64).resolution
    if alpha_max <= resolution:
        return np.full(n_alphas, resolution)
    return np.geomspace(alpha_max, alpha_max * eps, num=n_alphas)
