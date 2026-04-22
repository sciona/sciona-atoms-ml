"""Ghost witnesses for sklearn IterativeImputer helper atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_matrix(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    rows, cols = int(values.shape[0]), int(values.shape[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"{name} must be nonempty")
    return rows, cols


def _check_vector(values: AbstractArray, name: str) -> int:
    if len(values.shape) != 1:
        raise ValueError(f"{name} must be 1D")
    size = int(values.shape[0])
    if size < 1:
        raise ValueError(f"{name} must be nonempty")
    return size


def witness_iterative_ordered_feature_indices(
    mask_missing_values: AbstractArray,
    *,
    imputation_order: str = "ascending",
    skip_complete: bool = False,
    random_state: int = 0,
) -> AbstractArray:
    """Describe the feature index order for one imputation round."""
    del imputation_order, skip_complete, random_state
    _, n_features = _check_matrix(mask_missing_values, "mask_missing_values")
    return AbstractArray(shape=(n_features,), dtype="int64")


def witness_iterative_normalized_abs_corr_matrix(
    X_filled: AbstractArray,
    *,
    tolerance: float = 1e-6,
) -> AbstractArray:
    """Describe a normalized absolute-correlation matrix."""
    del tolerance
    _, n_features = _check_matrix(X_filled, "X_filled")
    if n_features < 2:
        raise ValueError("X_filled must have at least two features")
    return AbstractArray(shape=(n_features, n_features), dtype="float64")


def witness_iterative_neighbor_feature_indices(
    n_features: int,
    feat_idx: int,
    n_nearest_features: int | None,
    *,
    abs_corr_mat: AbstractArray | None = None,
    random_state: int = 0,
) -> AbstractArray:
    """Describe feature indices used to impute one target feature."""
    del random_state
    if n_features < 2:
        raise ValueError("n_features must be at least two")
    if feat_idx < 0 or feat_idx >= n_features:
        raise ValueError("feat_idx must reference a feature")
    if n_nearest_features is not None and n_nearest_features < n_features:
        if abs_corr_mat is None:
            raise ValueError("abs_corr_mat is required for neighbor sampling")
        if abs_corr_mat.shape != (n_features, n_features):
            raise ValueError("abs_corr_mat must be square")
        return AbstractArray(shape=(int(n_nearest_features),), dtype="int64")
    return AbstractArray(shape=(n_features - 1,), dtype="int64")


def witness_iterative_limit_vector(
    limit: float | AbstractArray | None,
    is_empty_feature: AbstractArray,
    *,
    limit_type: str,
    keep_empty_features: bool,
) -> AbstractArray:
    """Describe a scalar or vector bound expanded to feature bounds."""
    del limit, limit_type
    n_features = _check_vector(is_empty_feature, "is_empty_feature")
    return AbstractArray(shape=(n_features,), dtype="float64")


def witness_iterative_convergence_reached(
    X_current: AbstractArray,
    X_previous: AbstractArray,
    X_original: AbstractArray,
    mask_missing_values: AbstractArray,
    *,
    tol: float,
) -> bool:
    """Describe the early-stopping comparison for iterative imputation."""
    del tol
    shape = _check_matrix(X_current, "X_current")
    if _check_matrix(X_previous, "X_previous") != shape:
        raise ValueError("X_previous must match X_current")
    if _check_matrix(X_original, "X_original") != shape:
        raise ValueError("X_original must match X_current")
    if mask_missing_values.shape != shape:
        raise ValueError("mask_missing_values must match X_current")
    return False
