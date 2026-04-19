"""Ghost witnesses for sklearn VarianceThreshold atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray

from .state_models import VarianceThresholdState


def witness_variance_threshold_fit(
    X: AbstractArray,
    threshold: float = 0.0,
) -> AbstractArray:
    """Describe the per-feature variance vector learned during fitting."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[0] < 1:
        raise ValueError("X must contain at least one sample")
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    return AbstractArray(shape=(X.shape[1],), dtype="float64", min_val=0.0)


def witness_variance_threshold_support_mask(
    state: VarianceThresholdState,
) -> AbstractArray:
    """Describe the boolean feature support mask from fitted variances."""
    if state.n_features_in != state.variances.shape[0]:
        raise ValueError("state feature count must match variances")
    return AbstractArray(shape=(state.n_features_in,), dtype="bool")


def witness_variance_threshold_transform(
    X: AbstractArray,
    state: VarianceThresholdState,
) -> AbstractArray:
    """Describe column filtering with a fitted variance selector."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    selected = int((state.variances > state.threshold).sum())
    return AbstractArray(shape=(X.shape[0], selected), dtype=X.dtype)
