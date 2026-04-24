"""Deterministic IterativeImputer postprocessing atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy import stats

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_iterative_assign_feature_values,
    witness_iterative_clipped_imputed_values,
    witness_iterative_posterior_imputed_values,
    witness_iterative_restore_observed_values,
)


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


def _bool_vector(values: NDArray[np.bool_]) -> bool:
    array = np.asarray(values)
    return bool(array.ndim == 1 and array.shape[0] >= 1 and array.dtype == np.bool_)


def _bool_matrix(values: NDArray[np.bool_]) -> bool:
    array = np.asarray(values)
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and array.dtype == np.bool_)


def _numeric_bound(value: float) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and not np.isnan(float(value)))


def _bounds_valid(min_value: float, max_value: float) -> bool:
    return bool(_numeric_bound(min_value) and _numeric_bound(max_value) and float(min_value) <= float(max_value))


def _same_length(left: NDArray[np.float64], right: NDArray[np.float64]) -> bool:
    return bool(np.asarray(left).shape == np.asarray(right).shape)


def _assign_inputs_valid(
    X_filled: NDArray[np.float64],
    imputed_values: NDArray[np.float64],
    missing_row_mask: NDArray[np.bool_],
    feat_idx: int,
) -> bool:
    matrix = np.asarray(X_filled, dtype=np.float64)
    vector = np.asarray(imputed_values, dtype=np.float64)
    mask = np.asarray(missing_row_mask, dtype=np.bool_)
    return bool(
        _finite_matrix(matrix)
        and _finite_vector(vector)
        and _bool_vector(mask)
        and mask.shape[0] == matrix.shape[0]
        and isinstance(feat_idx, int)
        and not isinstance(feat_idx, bool)
        and 0 <= feat_idx < matrix.shape[1]
        and vector.shape[0] == int(np.sum(mask))
    )


def _restore_inputs_valid(
    X_target: NDArray[np.float64],
    X_source: NDArray[np.float64],
    observed_mask: NDArray[np.bool_],
) -> bool:
    target = np.asarray(X_target, dtype=np.float64)
    source = np.asarray(X_source, dtype=np.float64)
    mask = np.asarray(observed_mask)
    return bool(_finite_matrix(target) and _finite_matrix(source) and _bool_matrix(mask) and target.shape == source.shape == mask.shape)


@register_atom(witness_iterative_posterior_imputed_values)
@icontract.require(lambda mus: _finite_vector(mus), "mus must be a nonempty finite vector")
@icontract.require(lambda sigmas: _finite_vector(sigmas), "sigmas must be a nonempty finite vector")
@icontract.require(lambda mus, sigmas: _same_length(mus, sigmas), "mus and sigmas must have the same shape")
@icontract.require(lambda min_value, max_value: _bounds_valid(min_value, max_value), "min_value and max_value must be ordered numeric bounds")
@icontract.require(lambda random_state: isinstance(random_state, int) and not isinstance(random_state, bool) and random_state >= 0, "random_state must be a nonnegative integer")
@icontract.ensure(lambda result, mus: _finite_vector(result) and np.asarray(result).shape == np.asarray(mus).shape, "sampled values must be a finite vector matching mus")
def iterative_posterior_imputed_values(
    mus: NDArray[np.float64],
    sigmas: NDArray[np.float64],
    *,
    min_value: float,
    max_value: float,
    random_state: int = 0,
) -> NDArray[np.float64]:
    """Postprocess posterior mean and std vectors into bounded imputations."""
    means = np.asarray(mus, dtype=np.float64)
    scales = np.asarray(sigmas, dtype=np.float64)
    lower = float(min_value)
    upper = float(max_value)

    imputed_values = np.zeros(means.shape, dtype=np.float64)
    positive_sigmas = scales > 0.0
    imputed_values[~positive_sigmas] = means[~positive_sigmas]

    means_too_low = means < lower
    imputed_values[means_too_low] = lower

    means_too_high = means > upper
    imputed_values[means_too_high] = upper

    inrange_mask = positive_sigmas & ~means_too_low & ~means_too_high
    if np.any(inrange_mask):
        inrange_means = means[inrange_mask]
        inrange_sigmas = scales[inrange_mask]
        a = (lower - inrange_means) / inrange_sigmas
        b = (upper - inrange_means) / inrange_sigmas
        truncated_normal = stats.truncnorm(a=a, b=b, loc=inrange_means, scale=inrange_sigmas)
        imputed_values[inrange_mask] = truncated_normal.rvs(random_state=np.random.RandomState(int(random_state)))

    return np.asarray(imputed_values, dtype=np.float64)


@register_atom(witness_iterative_clipped_imputed_values)
@icontract.require(lambda predictions: _finite_vector(predictions), "predictions must be a nonempty finite vector")
@icontract.require(lambda min_value, max_value: _bounds_valid(min_value, max_value), "min_value and max_value must be ordered numeric bounds")
@icontract.ensure(lambda result, predictions: _finite_vector(result) and np.asarray(result).shape == np.asarray(predictions).shape, "clipped values must match the prediction shape")
def iterative_clipped_imputed_values(
    predictions: NDArray[np.float64],
    *,
    min_value: float,
    max_value: float,
) -> NDArray[np.float64]:
    """Clip deterministic one-feature predictions to IterativeImputer bounds."""
    return np.asarray(np.clip(np.asarray(predictions, dtype=np.float64), float(min_value), float(max_value)), dtype=np.float64)


@register_atom(witness_iterative_assign_feature_values)
@icontract.require(
    lambda X_filled, imputed_values, missing_row_mask, feat_idx: _assign_inputs_valid(X_filled, imputed_values, missing_row_mask, feat_idx),
    "X_filled, imputed_values, missing_row_mask, and feat_idx must describe a valid one-feature assignment",
)
@icontract.ensure(lambda result, X_filled: _finite_matrix(result) and np.asarray(result).shape == np.asarray(X_filled).shape, "updated matrix must stay finite and preserve shape")
def iterative_assign_feature_values(
    X_filled: NDArray[np.float64],
    imputed_values: NDArray[np.float64],
    missing_row_mask: NDArray[np.bool_],
    *,
    feat_idx: int,
) -> NDArray[np.float64]:
    """Assign one feature's imputed values into a dense filled matrix."""
    updated = np.array(X_filled, dtype=np.float64, copy=True)
    updated[np.asarray(missing_row_mask, dtype=np.bool_), int(feat_idx)] = np.asarray(imputed_values, dtype=np.float64)
    return np.asarray(updated, dtype=np.float64)


@register_atom(witness_iterative_restore_observed_values)
@icontract.require(
    lambda X_target, X_source, observed_mask: _restore_inputs_valid(X_target, X_source, observed_mask),
    "X_target, X_source, and observed_mask must be same-shaped dense arrays with a boolean mask",
)
@icontract.ensure(lambda result, X_target: _finite_matrix(result) and np.asarray(result).shape == np.asarray(X_target).shape, "restored matrix must stay finite and preserve shape")
def iterative_restore_observed_values(
    X_target: NDArray[np.float64],
    X_source: NDArray[np.float64],
    observed_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Restore observed entries from the original dense matrix."""
    restored = np.array(X_target, dtype=np.float64, copy=True)
    mask = np.asarray(observed_mask, dtype=np.bool_)
    source = np.asarray(X_source, dtype=np.float64)
    restored[mask] = source[mask]
    return np.asarray(restored, dtype=np.float64)
