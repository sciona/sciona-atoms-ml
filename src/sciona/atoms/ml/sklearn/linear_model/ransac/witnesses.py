"""Ghost witnesses for sklearn RANSAC helper atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_target(values: AbstractArray, name: str) -> int:
    if len(values.shape) not in {1, 2}:
        raise ValueError(f"{name} must be 1D or 2D")
    n_samples = int(values.shape[0])
    if n_samples < 1:
        raise ValueError(f"{name} must be nonempty")
    return n_samples


def _check_vector(values: AbstractArray, name: str) -> int:
    if len(values.shape) != 1:
        raise ValueError(f"{name} must be 1D")
    n_samples = int(values.shape[0])
    if n_samples < 1:
        raise ValueError(f"{name} must be nonempty")
    return n_samples


def witness_ransac_default_residual_threshold(y: AbstractArray) -> AbstractArray:
    """Describe the default residual threshold derived from target spread."""
    _check_target(y, "y")
    return AbstractArray(shape=(), dtype="float64")


def witness_ransac_loss_residuals(
    y_true: AbstractArray,
    y_pred: AbstractArray,
    *,
    loss: str = "absolute_error",
) -> AbstractArray:
    """Describe per-sample residual values for built-in loss choices."""
    if loss not in {"absolute_error", "squared_error"}:
        raise ValueError("loss must be a covered built-in option")
    n_samples = _check_target(y_true, "y_true")
    if _check_target(y_pred, "y_pred") != n_samples or y_pred.shape != y_true.shape:
        raise ValueError("targets and predictions must align")
    return AbstractArray(shape=(n_samples,), dtype="float64")


def witness_ransac_inlier_mask(
    residuals: AbstractArray,
    *,
    residual_threshold: float,
) -> AbstractArray:
    """Describe an inlier mask from residual values and a threshold."""
    del residual_threshold
    n_samples = _check_vector(residuals, "residuals")
    return AbstractArray(shape=(n_samples,), dtype="bool")


def witness_ransac_consensus_is_better(
    n_inliers: int,
    score: float,
    best_n_inliers: int,
    best_score: float,
) -> AbstractArray:
    """Describe whether a candidate consensus should replace the current best."""
    del n_inliers, score, best_n_inliers, best_score
    return AbstractArray(shape=(), dtype="bool")


def witness_ransac_dynamic_max_trials(
    n_inliers: int,
    n_samples: int,
    min_samples: int,
    probability: float,
) -> AbstractArray:
    """Describe the updated upper bound on random consensus trials."""
    del n_inliers, n_samples, min_samples, probability
    return AbstractArray(shape=(), dtype="float64")
