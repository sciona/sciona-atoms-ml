"""Ghost witnesses for SequentialFeatureSelector fit bookkeeping atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray

SequentialFeatureCountSpec = str | int | float


def witness_sequential_resolve_n_features_to_select(
    n_features: int,
    *,
    n_features_to_select: SequentialFeatureCountSpec = "auto",
    tol: float | None = None,
) -> AbstractArray:
    """Describe sklearn's resolved `n_features_to_select_` integer."""
    del n_features_to_select, tol
    if n_features < 2:
        raise ValueError("n_features must be at least 2")
    return AbstractArray(shape=(), dtype="int64", min_val=1.0, max_val=float(n_features - 1))


def witness_sequential_direction_tol_valid(
    *,
    direction: str = "forward",
    tol: float | None = None,
) -> AbstractArray:
    """Describe the direction-specific tolerance validity predicate."""
    del direction, tol
    return AbstractArray(shape=(), dtype="bool")


def witness_sequential_auto_select_enabled(
    *,
    n_features_to_select: SequentialFeatureCountSpec = "auto",
    tol: float | None = None,
) -> AbstractArray:
    """Describe sklearn's auto-selection early-stop mode flag."""
    del n_features_to_select, tol
    return AbstractArray(shape=(), dtype="bool")


def witness_sequential_iteration_count(
    n_features: int,
    n_features_to_select_resolved: int,
    *,
    n_features_to_select: SequentialFeatureCountSpec = "auto",
    direction: str = "forward",
) -> AbstractArray:
    """Describe sklearn's main-loop iteration count."""
    del n_features_to_select, direction
    if n_features < 2:
        raise ValueError("n_features must be at least 2")
    if not 1 <= n_features_to_select_resolved < n_features:
        raise ValueError("resolved feature count must lie in [1, n_features)")
    return AbstractArray(shape=(), dtype="int64", min_val=1.0, max_val=float(n_features - 1))


def witness_sequential_tolerance_break(
    old_score: float,
    new_score: float,
    *,
    tol: float,
) -> AbstractArray:
    """Describe whether the current score improvement falls below tolerance."""
    del old_score, new_score, tol
    return AbstractArray(shape=(), dtype="bool")


def witness_sequential_finalize_support(
    current_mask: AbstractArray,
    *,
    direction: str = "forward",
) -> AbstractArray:
    """Describe the finalized support mask after forward or backward selection."""
    del direction
    if len(current_mask.shape) != 1 or int(current_mask.shape[0]) < 1:
        raise ValueError("current_mask must be a nonempty 1D vector")
    return AbstractArray(shape=(int(current_mask.shape[0]),), dtype="bool")
