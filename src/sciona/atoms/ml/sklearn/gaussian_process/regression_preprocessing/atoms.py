"""Gaussian-process regression preprocessing atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_gp_regression_scaled_targets,
    witness_gp_regression_resolve_alpha,
    witness_gp_regression_target_count,
    witness_gp_regression_target_statistics,
    witness_gp_regression_validate_n_targets,
)

TargetStats = float | NDArray[np.float64]
AlphaLike = float | NDArray[np.float64]

def _finite_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim in {1, 2}
        and array.shape[0] >= 1
        and (array.ndim == 1 or array.shape[1] >= 1)
        and np.all(np.isfinite(array))
    )

def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)

def _target_count_valid(result: int, y: NDArray[np.float64]) -> bool:
    values = np.asarray(y, dtype=np.float64)
    expected = values.shape[1] if values.ndim == 2 else 1
    return bool(isinstance(result, int) and result == expected and result >= 1)

def _optional_positive_int(value: int | None) -> bool:
    return bool(value is None or _positive_int(value))

def _shape_stats(y: NDArray[np.float64]) -> int | tuple[int]:
    values = np.asarray(y, dtype=np.float64)
    return (values.shape[1],) if values.ndim == 2 else 1

def _target_stats_valid(value: TargetStats, shape: int | tuple[int]) -> bool:
    if isinstance(shape, int):
        return bool(np.isscalar(value) and np.isfinite(float(value)))
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.shape == shape and np.all(np.isfinite(array)))

def _target_scale_valid(value: TargetStats, shape: int | tuple[int]) -> bool:
    if not _target_stats_valid(value, shape):
        return False
    return bool(np.all(np.asarray(value, dtype=np.float64) > 0.0))

def _target_statistics_valid(result: tuple[TargetStats, TargetStats], y: NDArray[np.float64]) -> bool:
    if not isinstance(result, tuple) or len(result) != 2:
        return False
    mean, std = result
    shape = _shape_stats(y)
    return bool(_target_stats_valid(mean, shape) and _target_scale_valid(std, shape))

def _normalize_inputs_valid(y: NDArray[np.float64], y_train_mean: TargetStats, y_train_std: TargetStats) -> bool:
    if not _finite_matrix(y):
        return False
    shape = _shape_stats(y)
    return bool(_target_stats_valid(y_train_mean, shape) and _target_scale_valid(y_train_std, shape))

def _normalized_targets_valid(result: NDArray[np.float64], y: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    source = np.asarray(y, dtype=np.float64)
    return bool(values.shape == source.shape and np.all(np.isfinite(values)))

def _alpha_valid(alpha: AlphaLike) -> bool:
    if isinstance(alpha, bool):
        return False
    if np.isscalar(alpha):
        return bool(np.isfinite(float(alpha)) and float(alpha) >= 0.0)
    try:
        values = np.asarray(alpha, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 1 and values.shape[0] >= 1 and np.all(np.isfinite(values)) and np.all(values >= 0.0))

def _resolved_alpha_valid(result: AlphaLike, n_samples: int) -> bool:
    if np.isscalar(result):
        return bool(np.isfinite(float(result)) and float(result) >= 0.0)
    try:
        values = np.asarray(result, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 1 and values.shape == (n_samples,) and np.all(np.isfinite(values)) and np.all(values >= 0.0))

@register_atom(witness_gp_regression_target_count)
@icontract.require(lambda y: _finite_matrix(y), "y must be a nonempty finite vector or matrix")
@icontract.ensure(lambda result, y: _target_count_valid(result, y), "target count must match sklearn's observed output dimension")
def gp_regression_target_count(y: NDArray[np.float64]) -> int:
    """Count observed regression targets the way GaussianProcessRegressor.fit does."""
    values = np.asarray(y, dtype=np.float64)
    return int(values.shape[1] if values.ndim > 1 else 1)

@register_atom(witness_gp_regression_validate_n_targets)
@icontract.require(lambda observed_n_targets: _positive_int(observed_n_targets), "observed_n_targets must be a positive integer")
@icontract.require(lambda n_targets=None: _optional_positive_int(n_targets), "n_targets must be None or a positive integer")
@icontract.ensure(lambda result, observed_n_targets: isinstance(result, int) and result == observed_n_targets, "validated target count must preserve the observed count")
def gp_regression_validate_n_targets(
    observed_n_targets: int,
    *,
    n_targets: int | None = None,
) -> int:
    """Validate the observed target count against GaussianProcessRegressor.n_targets."""
    observed = int(observed_n_targets)
    if n_targets is not None and observed != int(n_targets):
        raise ValueError(
            "The number of targets seen in `y` is different from the parameter "
            f"`n_targets`. Got {observed} != {int(n_targets)}."
        )
    return observed

@register_atom(witness_gp_regression_target_statistics)
@icontract.require(lambda y: _finite_matrix(y), "y must be a nonempty finite vector or matrix")
@icontract.ensure(lambda result, y: _target_statistics_valid(result, y), "target statistics must match sklearn's mean and positive scale shape")
def gp_regression_target_statistics(
    y: NDArray[np.float64],
    *,
    normalize_y: bool = False,
) -> tuple[TargetStats, TargetStats]:
    from sklearn.preprocessing._data import _handle_zeros_in_scale
    """Compute the target mean and scale used by GaussianProcessRegressor.fit."""
    values = np.asarray(y, dtype=np.float64)
    if normalize_y:
        mean = np.mean(values, axis=0)
        std = _handle_zeros_in_scale(np.std(values, axis=0), copy=False)
        if values.ndim == 1:
            return float(mean), float(std)
        return np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)
    shape = _shape_stats(values)
    if isinstance(shape, int):
        return 0.0, 1.0
    return np.zeros(shape=shape, dtype=np.float64), np.ones(shape=shape, dtype=np.float64)

@register_atom(witness_gp_regression_scaled_targets)
@icontract.require(lambda y, y_train_mean, y_train_std: _normalize_inputs_valid(y, y_train_mean, y_train_std), "y, y_train_mean, and y_train_std must be finite and shape-compatible")
@icontract.ensure(lambda result, y: _normalized_targets_valid(result, y), "normalized targets must preserve y shape and remain finite")
def gp_regression_normalize_targets(
    y: NDArray[np.float64],
    y_train_mean: TargetStats,
    y_train_std: TargetStats,
) -> NDArray[np.float64]:
    """Normalize regression targets using GaussianProcessRegressor fit statistics."""
    values = np.asarray(y, dtype=np.float64)
    return np.asarray((values - y_train_mean) / y_train_std, dtype=np.float64)

@register_atom(witness_gp_regression_resolve_alpha)
@icontract.require(lambda alpha: _alpha_valid(alpha), "alpha must be a finite nonnegative scalar or nonempty finite nonnegative vector")
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be a positive integer")
@icontract.ensure(lambda result, n_samples: _resolved_alpha_valid(result, n_samples), "resolved alpha must be a finite nonnegative scalar or sample-length vector")
def gp_regression_resolve_alpha(
    alpha: AlphaLike,
    *,
    n_samples: int,
) -> AlphaLike:
    """Resolve GaussianProcessRegressor alpha to a usable scalar or sample-length vector."""
    if np.isscalar(alpha):
        return float(alpha)
    values = np.asarray(alpha, dtype=np.float64)
    if values.shape[0] != int(n_samples):
        if values.shape[0] == 1:
            return float(values[0])
        raise ValueError(
            "alpha must be a scalar or an array with same number of "
            f"entries as y. ({values.shape[0]} != {int(n_samples)})"
        )
    return np.asarray(values, dtype=np.float64)
