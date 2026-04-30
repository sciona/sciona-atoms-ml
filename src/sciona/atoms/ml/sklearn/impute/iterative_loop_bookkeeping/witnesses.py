"""Ghost witnesses for IterativeImputer loop bookkeeping helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_iterative_fit_initial_return_required(
    max_iter: int,
    mask_missing_values: AbstractArray,
) -> AbstractArray:
    """Describe the fit-time early-return predicate before the iterative loop."""
    if max_iter < 0:
        raise ValueError("max_iter must be nonnegative")
    if len(mask_missing_values.shape) != 2:
        raise ValueError("mask_missing_values must be 2D")
    return AbstractArray(shape=(), dtype="bool")


def witness_iterative_transform_initial_return_required(
    n_iter: int,
    mask_missing_values: AbstractArray,
) -> AbstractArray:
    """Describe the transform-time early-return predicate before replaying imputations."""
    if n_iter < 0:
        raise ValueError("n_iter must be nonnegative")
    if len(mask_missing_values.shape) != 2:
        raise ValueError("mask_missing_values must be 2D")
    return AbstractArray(shape=(), dtype="bool")


def witness_iterative_single_feature_return_required(
    n_features: int,
) -> AbstractArray:
    """Describe the single-feature early-return predicate."""
    if n_features < 0:
        raise ValueError("n_features must be nonnegative")
    return AbstractArray(shape=(), dtype="bool")


def witness_iterative_require_strict_limits(
    min_values: AbstractArray,
    max_values: AbstractArray,
) -> AbstractArray:
    """Describe validated strict min/max limit vectors."""
    if len(min_values.shape) != 1 or len(max_values.shape) != 1:
        raise ValueError("limit vectors must be 1D")
    if tuple(min_values.shape) != tuple(max_values.shape):
        raise ValueError("limit vectors must share shape")
    return AbstractArray(shape=(int(min_values.shape[0]),), dtype="float64")


def witness_iterative_missing_feature_count(
    ordered_idx: AbstractArray,
) -> AbstractArray:
    """Describe the scalar feature-with-missing count from ordered indices."""
    if len(ordered_idx.shape) != 1:
        raise ValueError("ordered_idx must be 1D")
    return AbstractArray(shape=(), dtype="int64", min_val=0.0)


def witness_iterative_normalized_tolerance(
    X_original: AbstractArray,
    mask_missing_values: AbstractArray,
    *,
    tol: float,
) -> AbstractArray:
    """Describe sklearn's scaled tolerance before iterative convergence checks."""
    del tol
    if len(X_original.shape) != 2 or len(mask_missing_values.shape) != 2:
        raise ValueError("inputs must be 2D")
    if tuple(X_original.shape) != tuple(mask_missing_values.shape):
        raise ValueError("inputs must share shape")
    return AbstractArray(shape=(), dtype="float64", min_val=0.0)


def witness_iterative_imputations_per_round(
    sequence_length: int,
    n_iter: int,
) -> AbstractArray:
    """Describe sklearn's transform-time imputation grouping size."""
    if sequence_length < 0 or n_iter < 1:
        raise ValueError("sequence_length must be nonnegative and n_iter must be positive")
    return AbstractArray(shape=(), dtype="int64", min_val=0.0)

