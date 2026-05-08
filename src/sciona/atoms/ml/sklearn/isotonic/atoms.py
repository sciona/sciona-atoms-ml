"""Isotonic regression atoms adapted from scikit-learn."""

from __future__ import annotations

import math
import warnings

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy import optimize
from scipy.stats import spearmanr

from sciona.ghost.registry import register_atom

from .state_models import IsotonicRegressionState
from .witnesses import (
    witness_isotonic_regression,
    witness_isotonic_regression_fit,
    witness_isotonic_regression_predict,
    witness_isotonic_regression_transform,
)

def _is_vector(y: NDArray[np.float64]) -> bool:
    return bool(np.asarray(y).ndim == 1)

def _valid_x_shape(X: NDArray[np.float64]) -> bool:
    array = np.asarray(X)
    return bool(array.ndim == 1 or (array.ndim == 2 and array.shape[1] == 1))

def _same_length(a: NDArray[np.float64], b: NDArray[np.float64]) -> bool:
    return int(np.asarray(a).shape[0]) == int(np.asarray(b).shape[0])

def _valid_increasing(increasing: bool | str) -> bool:
    return increasing in {True, False, "auto"}

def _valid_out_of_bounds(out_of_bounds: str) -> bool:
    return out_of_bounds in {"nan", "clip", "raise"}

def _monotonic(values: NDArray[np.float64], increasing: bool) -> bool:
    diffs = np.diff(values)
    return bool(np.all(diffs >= -1e-12) if increasing else np.all(diffs <= 1e-12))

def _threshold_state_valid(state: IsotonicRegressionState) -> bool:
    if state.x_thresholds.ndim != 1 or state.y_thresholds.ndim != 1:
        return False
    if state.x_thresholds.shape != state.y_thresholds.shape or state.x_thresholds.size < 1:
        return False
    if not np.all(np.diff(state.x_thresholds) > 0):
        return state.x_thresholds.size == 1
    return _monotonic(state.y_thresholds, state.increasing)

def _check_increasing(x: NDArray[np.float64], y: NDArray[np.float64]) -> bool:
    rho, _ = spearmanr(x, y)
    increasing_bool = rho >= 0
    if rho not in [-1.0, 1.0] and len(x) > 3:
        fisher = 0.5 * math.log((1.0 + rho) / (1.0 - rho))
        fisher_se = 1 / math.sqrt(len(x) - 3)
        rho_0 = math.tanh(fisher - 1.96 * fisher_se)
        rho_1 = math.tanh(fisher + 1.96 * fisher_se)
        if np.sign(rho_0) != np.sign(rho_1):
            warnings.warn(
                "Confidence interval of the Spearman correlation coefficient spans zero. "
                "Determination of ``increasing`` may be suspect.",
                UserWarning,
                stacklevel=2,
            )
    return bool(increasing_bool)

def _make_unique(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    sample_weight: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    unique_x, inverse = np.unique(X, return_inverse=True)
    weight_sums = np.bincount(inverse, weights=sample_weight)
    y_sums = np.bincount(inverse, weights=y * sample_weight)
    return unique_x, y_sums / weight_sums, weight_sums

def _prepare_bounds(y_min: float | None, y_max: float | None) -> tuple[float | None, float | None]:
    if y_min is not None or y_max is not None:
        return (-np.inf if y_min is None else y_min, np.inf if y_max is None else y_max)
    return y_min, y_max

@register_atom(witness_isotonic_regression)
@icontract.require(lambda y: _is_vector(y), "y must be a 1D vector")
@icontract.require(lambda y, sample_weight: sample_weight is None or _same_length(y, sample_weight), "sample_weight must match y length")
@icontract.ensure(lambda result, y: result.shape == np.asarray(y).shape, "isotonic result must match y shape")
@icontract.ensure(lambda result, increasing: _monotonic(result, increasing), "isotonic result must be monotonic")
def isotonic_regression(
    y: NDArray[np.float64],
    *,
    sample_weight: NDArray[np.float64] | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    increasing: bool = True,
) -> NDArray[np.float64]:
    from sklearn.utils import check_array, check_consistent_length
    """Solve one-dimensional isotonic regression for a response vector."""
    checked_y = check_array(y, ensure_2d=False, input_name="y", dtype=[np.float64, np.float32])
    result = optimize.isotonic_regression(y=checked_y, weights=sample_weight, increasing=increasing)
    fitted = np.asarray(result.x, dtype=checked_y.dtype)
    clip_min, clip_max = _prepare_bounds(y_min, y_max)
    if clip_min is not None or clip_max is not None:
        np.clip(fitted, clip_min, clip_max, fitted)
    return fitted

@register_atom(witness_isotonic_regression_fit)
@icontract.require(lambda X: _valid_x_shape(X), "X must be 1D or 2D with one feature")
@icontract.require(lambda y: _is_vector(y), "y must be a 1D vector")
@icontract.require(lambda X, y: _same_length(X, y), "X and y must have equal sample count")
@icontract.require(lambda X, sample_weight: sample_weight is None or _same_length(X, sample_weight), "sample_weight must match X length")
@icontract.require(lambda increasing: _valid_increasing(increasing), "increasing must be boolean or 'auto'")
@icontract.require(lambda out_of_bounds: _valid_out_of_bounds(out_of_bounds), "out_of_bounds must be 'nan', 'clip', or 'raise'")
@icontract.ensure(lambda result: _threshold_state_valid(result), "threshold state must be sorted and monotonic")
def isotonic_regression_fit(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    sample_weight: NDArray[np.float64] | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    increasing: bool | str = True,
    out_of_bounds: str = "nan",
) -> IsotonicRegressionState:
    from sklearn.utils import check_array, check_consistent_length
    from sklearn.utils.validation import _check_sample_weight
    """Fit isotonic regression thresholds for later interpolation."""
    checked_x = check_array(X, input_name="X", dtype=[np.float64, np.float32], accept_sparse=False, ensure_2d=False)
    checked_y = check_array(y, input_name="y", dtype=checked_x.dtype, accept_sparse=False, ensure_2d=False)
    check_consistent_length(checked_x, checked_y, sample_weight)
    if not _valid_x_shape(checked_x):
        raise ValueError("Isotonic regression input X should be a 1d array or 2d array with 1 feature")
    checked_x = checked_x.reshape(-1)

    resolved_increasing = _check_increasing(checked_x, checked_y) if increasing == "auto" else bool(increasing)

    checked_weight = _check_sample_weight(sample_weight, checked_x, dtype=checked_x.dtype)
    mask = checked_weight > 0
    checked_x, checked_y, checked_weight = checked_x[mask], checked_y[mask], checked_weight[mask]

    order = np.lexsort((checked_y, checked_x))
    sorted_x, sorted_y, sorted_weight = [array[order] for array in (checked_x, checked_y, checked_weight)]
    unique_x, unique_y, unique_weight = _make_unique(sorted_x, sorted_y, sorted_weight)

    fitted_y = isotonic_regression(
        unique_y,
        sample_weight=unique_weight,
        y_min=y_min,
        y_max=y_max,
        increasing=resolved_increasing,
    )
    x_min = float(np.min(unique_x))
    x_max = float(np.max(unique_x))

    keep_data = np.ones((len(fitted_y),), dtype=bool)
    keep_data[1:-1] = np.logical_or(
        np.not_equal(fitted_y[1:-1], fitted_y[:-2]),
        np.not_equal(fitted_y[1:-1], fitted_y[2:]),
    )
    x_thresholds = np.asarray(unique_x[keep_data], dtype=checked_x.dtype)
    y_thresholds = np.asarray(fitted_y[keep_data], dtype=checked_x.dtype)
    return IsotonicRegressionState(
        x_thresholds=x_thresholds,
        y_thresholds=y_thresholds,
        x_min=x_min,
        x_max=x_max,
        increasing=resolved_increasing,
        out_of_bounds=out_of_bounds,
    )

@register_atom(witness_isotonic_regression_transform)
@icontract.require(lambda T: _valid_x_shape(T), "T must be 1D or 2D with one feature")
@icontract.require(lambda state: _threshold_state_valid(state), "state thresholds must be sorted and monotonic")
@icontract.ensure(lambda result, T: result.shape == (np.asarray(T).shape[0],), "transform output must have one value per sample")
def isotonic_regression_transform(
    T: NDArray[np.float64],
    state: IsotonicRegressionState,
) -> NDArray[np.float64]:
    from sklearn.utils import check_array, check_consistent_length
    """Interpolate new samples with fitted isotonic regression thresholds."""
    checked_t = check_array(T, dtype=state.x_thresholds.dtype, ensure_2d=False)
    if not _valid_x_shape(checked_t):
        raise ValueError("Isotonic regression input X should be a 1d array or 2d array with 1 feature")
    flat_t = checked_t.reshape(-1)
    if state.y_thresholds.size == 1:
        return np.repeat(state.y_thresholds[0], flat_t.shape).astype(flat_t.dtype)
    if state.out_of_bounds == "raise" and (np.min(flat_t) < state.x_min or np.max(flat_t) > state.x_max):
        raise ValueError("A value in x_new is below the interpolation range's minimum value or above the maximum value.")
    if state.out_of_bounds == "clip":
        flat_t = np.clip(flat_t, state.x_min, state.x_max)
        left = None
        right = None
    else:
        left = np.nan
        right = np.nan
    result = np.interp(flat_t, state.x_thresholds, state.y_thresholds, left=left, right=right)
    return np.asarray(result, dtype=flat_t.dtype)

@register_atom(witness_isotonic_regression_predict)
@icontract.require(lambda T: _valid_x_shape(T), "T must be 1D or 2D with one feature")
@icontract.require(lambda state: _threshold_state_valid(state), "state thresholds must be sorted and monotonic")
@icontract.ensure(lambda result, T: result.shape == (np.asarray(T).shape[0],), "predict output must have one value per sample")
def isotonic_regression_predict(
    T: NDArray[np.float64],
    state: IsotonicRegressionState,
) -> NDArray[np.float64]:
    """Predict new samples with fitted isotonic regression thresholds."""
    return isotonic_regression_transform(T, state)
