"""Ghost witnesses for sklearn IterativeImputer initial-bookkeeping atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_bool_matrix(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    rows, cols = int(values.shape[0]), int(values.shape[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"{name} must be nonempty")
    return rows, cols


def _check_float_matrix_allow_empty_columns(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    rows, cols = int(values.shape[0]), int(values.shape[1])
    if rows < 1 or cols < 0:
        raise ValueError(f"{name} must have at least one row")
    return rows, cols


def _check_bool_vector(values: AbstractArray, name: str) -> int:
    if len(values.shape) != 1:
        raise ValueError(f"{name} must be 1D")
    size = int(values.shape[0])
    if size < 1:
        raise ValueError(f"{name} must be nonempty")
    return size


def witness_iterative_empty_feature_mask(mask_missing_values: AbstractArray) -> AbstractArray:
    """Describe the featurewise all-missing mask seen during fit."""
    _, cols = _check_bool_matrix(mask_missing_values, "mask_missing_values")
    return AbstractArray(shape=(cols,), dtype="bool")


def witness_iterative_filter_nonempty_matrix(
    values: AbstractArray,
    is_empty_feature: AbstractArray,
) -> AbstractArray:
    """Describe dropping empty-feature columns from a dense matrix."""
    rows, cols = _check_float_matrix_allow_empty_columns(values, "values")
    if _check_bool_vector(is_empty_feature, "is_empty_feature") != cols:
        raise ValueError("is_empty_feature must match matrix width")
    return AbstractArray(shape=(rows, cols), dtype="float64")


def witness_iterative_filter_nonempty_missing_mask(
    mask_missing_values: AbstractArray,
    is_empty_feature: AbstractArray,
) -> AbstractArray:
    """Describe dropping empty-feature columns from a missing mask."""
    rows, cols = _check_bool_matrix(mask_missing_values, "mask_missing_values")
    if _check_bool_vector(is_empty_feature, "is_empty_feature") != cols:
        raise ValueError("is_empty_feature must match matrix width")
    return AbstractArray(shape=(rows, cols), dtype="bool")


def witness_iterative_clear_empty_feature_missing_mask(
    mask_missing_values: AbstractArray,
    is_empty_feature: AbstractArray,
) -> AbstractArray:
    """Describe clearing missing flags for kept empty features."""
    shape = _check_bool_matrix(mask_missing_values, "mask_missing_values")
    if _check_bool_vector(is_empty_feature, "is_empty_feature") != shape[1]:
        raise ValueError("is_empty_feature must match matrix width")
    return AbstractArray(shape=shape, dtype="bool")


def witness_iterative_restore_empty_feature_imputations(
    X: AbstractArray,
    X_filled: AbstractArray,
    is_empty_feature: AbstractArray,
) -> AbstractArray:
    """Describe copying imputed values into empty-feature columns."""
    shape = _check_float_matrix_allow_empty_columns(X, "X")
    if _check_float_matrix_allow_empty_columns(X_filled, "X_filled") != shape:
        raise ValueError("X_filled must match X")
    if _check_bool_vector(is_empty_feature, "is_empty_feature") != shape[1]:
        raise ValueError("is_empty_feature must match matrix width")
    return AbstractArray(shape=shape, dtype="float64")
