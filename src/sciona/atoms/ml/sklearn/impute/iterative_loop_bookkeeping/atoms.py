"""IterativeImputer loop bookkeeping helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_iterative_fit_initial_return_required,
    witness_iterative_imputations_per_round,
    witness_iterative_missing_feature_count,
    witness_iterative_normalized_tolerance,
    witness_iterative_require_strict_limits,
    witness_iterative_single_feature_return_required,
    witness_iterative_transform_initial_return_required,
)


def _bool_matrix(values: object) -> bool:
    array = np.asarray(values)
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and array.dtype == np.bool_)


def _finite_matrix_with_missing(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 2
        and array.shape[0] >= 1
        and array.shape[1] >= 1
        and np.all(np.isfinite(array) | np.isnan(array))
    )


def _nonnegative_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 0)


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _finite_positive(value: object) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(float(value))
        and float(value) > 0.0
    )


def _finite_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array) | np.isinf(array)))


def _int_vector(values: object) -> bool:
    array = np.asarray(values)
    return bool(array.ndim == 1 and array.dtype.kind in {"i", "u"})


def _same_shape(left: object, right: object) -> bool:
    return bool(np.asarray(left).shape == np.asarray(right).shape)


@register_atom(witness_iterative_fit_initial_return_required)
@icontract.require(lambda max_iter: _nonnegative_int(max_iter), "max_iter must be a nonnegative integer")
@icontract.require(lambda mask_missing_values: _bool_matrix(mask_missing_values), "mask_missing_values must be a nonempty boolean matrix")
@icontract.ensure(lambda result: isinstance(result, bool), "fit early-return predicate must be boolean")
def iterative_fit_initial_return_required(
    max_iter: int,
    mask_missing_values: NDArray[np.bool_],
) -> bool:
    """Apply sklearn's fit-time early-return predicate before the iterative loop."""
    return bool(int(max_iter) == 0 or np.all(np.asarray(mask_missing_values, dtype=np.bool_)))


@register_atom(witness_iterative_transform_initial_return_required)
@icontract.require(lambda n_iter: _nonnegative_int(n_iter), "n_iter must be a nonnegative integer")
@icontract.require(lambda mask_missing_values: _bool_matrix(mask_missing_values), "mask_missing_values must be a nonempty boolean matrix")
@icontract.ensure(lambda result: isinstance(result, bool), "transform early-return predicate must be boolean")
def iterative_transform_initial_return_required(
    n_iter: int,
    mask_missing_values: NDArray[np.bool_],
) -> bool:
    """Apply sklearn's transform-time early-return predicate before replaying imputations."""
    return bool(int(n_iter) == 0 or np.all(np.asarray(mask_missing_values, dtype=np.bool_)))


@register_atom(witness_iterative_single_feature_return_required)
@icontract.require(lambda n_features: _nonnegative_int(n_features), "n_features must be a nonnegative integer")
@icontract.ensure(lambda result: isinstance(result, bool), "single-feature predicate must be boolean")
def iterative_single_feature_return_required(n_features: int) -> bool:
    """Apply sklearn's single-feature early-return predicate."""
    return bool(int(n_features) == 1)


@register_atom(witness_iterative_require_strict_limits)
@icontract.require(lambda min_values: _finite_vector(min_values), "min_values must be a finite 1D vector allowing infinities")
@icontract.require(lambda max_values: _finite_vector(max_values), "max_values must be a finite 1D vector allowing infinities")
@icontract.require(lambda min_values, max_values: _same_shape(min_values, max_values), "limit vectors must share shape")
@icontract.ensure(lambda result, min_values: np.asarray(result, dtype=np.float64).shape == np.asarray(min_values).shape, "validated limit vector must preserve shape")
def iterative_require_strict_limits(
    min_values: NDArray[np.float64],
    max_values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Require sklearn's strict elementwise max_value > min_value condition."""
    lower = np.asarray(min_values, dtype=np.float64)
    upper = np.asarray(max_values, dtype=np.float64)
    if not np.all(np.greater(upper, lower)):
        raise ValueError("One (or more) features have min_value >= max_value.")
    return np.asarray(upper, dtype=np.float64)


@register_atom(witness_iterative_missing_feature_count)
@icontract.require(lambda ordered_idx: _int_vector(ordered_idx), "ordered_idx must be a 1D integer vector")
@icontract.ensure(lambda result: _nonnegative_int(result), "missing-feature count must be a nonnegative integer")
def iterative_missing_feature_count(ordered_idx: NDArray[np.int64]) -> int:
    """Count sklearn's features-with-missing summary from ordered indices."""
    return int(np.asarray(ordered_idx).shape[0])


@register_atom(witness_iterative_normalized_tolerance)
@icontract.require(lambda X_original: _finite_matrix_with_missing(X_original), "X_original must be finite apart from missing entries")
@icontract.require(lambda mask_missing_values: _bool_matrix(mask_missing_values), "mask_missing_values must be a nonempty boolean matrix")
@icontract.require(lambda X_original, mask_missing_values: _same_shape(X_original, mask_missing_values), "X_original and mask_missing_values must share shape")
@icontract.require(lambda tol: _finite_positive(tol), "tol must be a finite positive scalar")
@icontract.ensure(lambda result: isinstance(result, float) and np.isfinite(result) and result >= 0.0, "normalized tolerance must be finite and nonnegative")
def iterative_normalized_tolerance(
    X_original: NDArray[np.float64],
    mask_missing_values: NDArray[np.bool_],
    *,
    tol: float,
) -> float:
    """Compute sklearn's scaled tolerance before iterative convergence checks."""
    original = np.asarray(X_original, dtype=np.float64)
    missing = np.asarray(mask_missing_values, dtype=np.bool_)
    observed = np.abs(original[~missing])
    return float(float(tol) * float(np.max(observed)))


@register_atom(witness_iterative_imputations_per_round)
@icontract.require(lambda sequence_length: _nonnegative_int(sequence_length), "sequence_length must be a nonnegative integer")
@icontract.require(lambda n_iter: _positive_int(n_iter), "n_iter must be a positive integer")
@icontract.require(lambda sequence_length, n_iter: int(sequence_length) % int(n_iter) == 0, "sequence_length must divide evenly across rounds")
@icontract.ensure(lambda result: _nonnegative_int(result), "imputations-per-round must be a nonnegative integer")
def iterative_imputations_per_round(sequence_length: int, n_iter: int) -> int:
    """Compute sklearn's transform-time imputation grouping size."""
    return int(int(sequence_length) // int(n_iter))

