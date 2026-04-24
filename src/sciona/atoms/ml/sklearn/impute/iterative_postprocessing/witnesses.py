"""Ghost witnesses for sklearn IterativeImputer postprocessing helper atoms."""

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


def witness_iterative_posterior_imputed_values(
    mus: AbstractArray,
    sigmas: AbstractArray,
    *,
    min_value: float,
    max_value: float,
    random_state: int = 0,
) -> AbstractArray:
    """Describe posterior-sampled imputations for one target feature."""
    del min_value, max_value, random_state
    shape = _check_vector(mus, "mus")
    if _check_vector(sigmas, "sigmas") != shape:
        raise ValueError("sigmas must match mus")
    return AbstractArray(shape=(shape,), dtype="float64")


def witness_iterative_clipped_imputed_values(
    predictions: AbstractArray,
    *,
    min_value: float,
    max_value: float,
) -> AbstractArray:
    """Describe clipped deterministic imputations for one target feature."""
    del min_value, max_value
    size = _check_vector(predictions, "predictions")
    return AbstractArray(shape=(size,), dtype="float64")


def witness_iterative_assign_feature_values(
    X_filled: AbstractArray,
    imputed_values: AbstractArray,
    missing_row_mask: AbstractArray,
    *,
    feat_idx: int,
) -> AbstractArray:
    """Describe assigning one feature's imputed values into a dense matrix."""
    rows, cols = _check_matrix(X_filled, "X_filled")
    if _check_vector(missing_row_mask, "missing_row_mask") != rows:
        raise ValueError("missing_row_mask must have one entry per row")
    _check_vector(imputed_values, "imputed_values")
    if feat_idx < 0 or feat_idx >= cols:
        raise ValueError("feat_idx must reference a feature")
    return AbstractArray(shape=(rows, cols), dtype="float64")


def witness_iterative_restore_observed_values(
    X_target: AbstractArray,
    X_source: AbstractArray,
    observed_mask: AbstractArray,
) -> AbstractArray:
    """Describe restoring observed values into a dense imputed matrix."""
    shape = _check_matrix(X_target, "X_target")
    if _check_matrix(X_source, "X_source") != shape:
        raise ValueError("X_source must match X_target")
    if observed_mask.shape != shape:
        raise ValueError("observed_mask must match X_target")
    return AbstractArray(shape=shape, dtype="float64")
