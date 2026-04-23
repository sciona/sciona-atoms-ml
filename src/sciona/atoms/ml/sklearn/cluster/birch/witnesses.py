"""Ghost witnesses for BIRCH no-global-clustering atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray

from .state_models import BirchNoGlobalState


def _check_2d(X: AbstractArray) -> tuple[int, int]:
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    rows = int(X.shape[0])
    cols = int(X.shape[1])
    if rows < 1 or cols < 1:
        raise ValueError("X must have at least one sample and one feature")
    return rows, cols


def _check_birch_options(
    X: AbstractArray,
    threshold: float,
    branching_factor: int,
    compute_labels: bool,
) -> tuple[int, int]:
    del compute_labels
    n_samples, n_features = _check_2d(X)
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")
    if branching_factor < 2:
        raise ValueError("branching_factor must be at least two")
    return n_samples, n_features


def witness_birch_fit_no_global(
    X: AbstractArray,
    *,
    threshold: float = 0.5,
    branching_factor: int = 50,
    compute_labels: bool = True,
) -> AbstractArray:
    """Describe BIRCH no-global subcluster centers."""
    n_samples, n_features = _check_birch_options(X, threshold, branching_factor, compute_labels)
    return AbstractArray(shape=(n_samples, n_features), dtype="float64")


def witness_birch_predict_no_global(
    X: AbstractArray,
    state: BirchNoGlobalState,
) -> AbstractArray:
    """Describe nearest-subcluster BIRCH labels."""
    n_samples, n_features = _check_2d(X)
    if n_features != state.n_features_in:
        raise ValueError("X feature count must match BIRCH state")
    return AbstractArray(shape=(n_samples,), dtype="int64", min_val=0)


def witness_birch_transform_no_global(
    X: AbstractArray,
    state: BirchNoGlobalState,
) -> AbstractArray:
    """Describe distances from samples to BIRCH subcluster centers."""
    n_samples, n_features = _check_2d(X)
    if n_features != state.n_features_in:
        raise ValueError("X feature count must match BIRCH state")
    return AbstractArray(shape=(n_samples, state.n_features_out), dtype="float64", min_val=0.0)
