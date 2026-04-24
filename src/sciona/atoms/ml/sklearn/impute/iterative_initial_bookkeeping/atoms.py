"""Deterministic IterativeImputer initial-bookkeeping atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_iterative_clear_empty_feature_missing_mask,
    witness_iterative_empty_feature_mask,
    witness_iterative_filter_nonempty_matrix,
    witness_iterative_filter_nonempty_missing_mask,
    witness_iterative_restore_empty_feature_imputations,
)


def _bool_matrix(values: NDArray[np.bool_]) -> bool:
    array = np.asarray(values)
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and array.dtype == np.bool_)


def _bool_vector(values: NDArray[np.bool_]) -> bool:
    array = np.asarray(values)
    return bool(array.ndim == 1 and array.shape[0] >= 1 and array.dtype == np.bool_)


def _matrix_with_missing(values: NDArray[np.float64]) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and np.all(np.isfinite(array) | np.isnan(array)))


def _finite_matrix(values: NDArray[np.float64]) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _matching_feature_vector(values: NDArray[np.float64] | NDArray[np.bool_], is_empty_feature: NDArray[np.bool_]) -> bool:
    return bool(_bool_vector(is_empty_feature) and np.asarray(values).ndim == 2 and np.asarray(values).shape[1] == np.asarray(is_empty_feature).shape[0])


def _filtered_matrix_result_valid(result: NDArray[np.float64], values: NDArray[np.float64], is_empty_feature: NDArray[np.bool_]) -> bool:
    filtered = np.asarray(result, dtype=np.float64)
    original = np.asarray(values, dtype=np.float64)
    empty = np.asarray(is_empty_feature, dtype=np.bool_)
    return bool(filtered.shape == (original.shape[0], int(np.sum(~empty))) and np.array_equal(filtered, original[:, ~empty], equal_nan=True))


def _filtered_mask_result_valid(result: NDArray[np.bool_], mask_missing_values: NDArray[np.bool_], is_empty_feature: NDArray[np.bool_]) -> bool:
    filtered = np.asarray(result, dtype=np.bool_)
    mask = np.asarray(mask_missing_values, dtype=np.bool_)
    empty = np.asarray(is_empty_feature, dtype=np.bool_)
    return bool(filtered.shape == (mask.shape[0], int(np.sum(~empty))) and np.array_equal(filtered, mask[:, ~empty]))


def _restored_result_valid(result: NDArray[np.float64], X: NDArray[np.float64], X_filled: NDArray[np.float64], is_empty_feature: NDArray[np.bool_]) -> bool:
    restored = np.asarray(result, dtype=np.float64)
    original = np.asarray(X, dtype=np.float64)
    filled = np.asarray(X_filled, dtype=np.float64)
    empty = np.asarray(is_empty_feature, dtype=np.bool_)
    expected = np.array(original, dtype=np.float64, copy=True)
    expected[:, empty] = filled[:, empty]
    return bool(restored.shape == original.shape and np.array_equal(restored, expected, equal_nan=True))


@register_atom(witness_iterative_empty_feature_mask)
@icontract.require(lambda mask_missing_values: _bool_matrix(mask_missing_values), "mask_missing_values must be a nonempty boolean matrix")
@icontract.ensure(lambda result, mask_missing_values: _bool_vector(result) and np.asarray(result).shape == (np.asarray(mask_missing_values).shape[1],), "empty-feature mask must be one boolean per feature")
def iterative_empty_feature_mask(mask_missing_values: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Detect features that are entirely missing in the fit-time mask."""
    return np.all(np.asarray(mask_missing_values, dtype=np.bool_), axis=0)


@register_atom(witness_iterative_filter_nonempty_matrix)
@icontract.require(lambda values: _matrix_with_missing(values), "values must be a dense matrix with finite values or NaNs")
@icontract.require(lambda values, is_empty_feature: _matching_feature_vector(values, is_empty_feature), "is_empty_feature must match the matrix width")
@icontract.ensure(lambda result, values, is_empty_feature: _filtered_matrix_result_valid(result, values, is_empty_feature), "filtered matrix must keep exactly the nonempty columns")
def iterative_filter_nonempty_matrix(
    values: NDArray[np.float64],
    is_empty_feature: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Drop empty-feature columns from a dense matrix."""
    matrix = np.asarray(values, dtype=np.float64)
    empty = np.asarray(is_empty_feature, dtype=np.bool_)
    return np.asarray(matrix[:, ~empty], dtype=np.float64)


@register_atom(witness_iterative_filter_nonempty_missing_mask)
@icontract.require(lambda mask_missing_values: _bool_matrix(mask_missing_values), "mask_missing_values must be a nonempty boolean matrix")
@icontract.require(lambda mask_missing_values, is_empty_feature: _matching_feature_vector(mask_missing_values, is_empty_feature), "is_empty_feature must match the mask width")
@icontract.ensure(lambda result, mask_missing_values, is_empty_feature: _filtered_mask_result_valid(result, mask_missing_values, is_empty_feature), "filtered mask must keep exactly the nonempty columns")
def iterative_filter_nonempty_missing_mask(
    mask_missing_values: NDArray[np.bool_],
    is_empty_feature: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Drop empty-feature columns from a missing-value mask."""
    mask = np.asarray(mask_missing_values, dtype=np.bool_)
    empty = np.asarray(is_empty_feature, dtype=np.bool_)
    return np.asarray(mask[:, ~empty], dtype=np.bool_)


@register_atom(witness_iterative_clear_empty_feature_missing_mask)
@icontract.require(lambda mask_missing_values: _bool_matrix(mask_missing_values), "mask_missing_values must be a nonempty boolean matrix")
@icontract.require(lambda mask_missing_values, is_empty_feature: _matching_feature_vector(mask_missing_values, is_empty_feature), "is_empty_feature must match the mask width")
@icontract.ensure(lambda result, mask_missing_values: _bool_matrix(result) and np.asarray(result).shape == np.asarray(mask_missing_values).shape, "cleared mask must preserve shape")
def iterative_clear_empty_feature_missing_mask(
    mask_missing_values: NDArray[np.bool_],
    is_empty_feature: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Clear missing flags for empty features kept in the working matrix."""
    cleared = np.array(mask_missing_values, dtype=np.bool_, copy=True)
    cleared[:, np.asarray(is_empty_feature, dtype=np.bool_)] = False
    return np.asarray(cleared, dtype=np.bool_)


@register_atom(witness_iterative_restore_empty_feature_imputations)
@icontract.require(lambda X: _matrix_with_missing(X), "X must be a dense matrix with finite values or NaNs")
@icontract.require(lambda X_filled: _finite_matrix(X_filled), "X_filled must be a dense finite matrix")
@icontract.require(lambda X, X_filled: np.asarray(X).shape == np.asarray(X_filled).shape, "X and X_filled must share shape")
@icontract.require(lambda X, is_empty_feature: _matching_feature_vector(X, is_empty_feature), "is_empty_feature must match the matrix width")
@icontract.ensure(lambda result, X, X_filled, is_empty_feature: _restored_result_valid(result, X, X_filled, is_empty_feature), "restored matrix must copy filled values into exactly the empty columns")
def iterative_restore_empty_feature_imputations(
    X: NDArray[np.float64],
    X_filled: NDArray[np.float64],
    is_empty_feature: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Copy imputed values into the empty-feature columns of the original matrix."""
    restored = np.array(X, dtype=np.float64, copy=True)
    empty = np.asarray(is_empty_feature, dtype=np.bool_)
    restored[:, empty] = np.asarray(X_filled, dtype=np.float64)[:, empty]
    return np.asarray(restored, dtype=np.float64)
