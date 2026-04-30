"""Ghost witnesses for BIRCH subcluster math helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_birch_subcluster_singleton(linear_sum: AbstractArray) -> AbstractArray:
    """Describe singleton BIRCH subcluster statistics from one feature vector."""
    if len(linear_sum.shape) != 1 or int(linear_sum.shape[0]) < 1:
        raise ValueError("linear_sum must be a nonempty 1D vector")
    return AbstractArray(shape=(), dtype="object")


def witness_birch_subcluster_update(base_state: object, added_state: object) -> AbstractArray:
    """Describe updated BIRCH subcluster statistics after deterministic accumulation."""
    del base_state, added_state
    return AbstractArray(shape=(), dtype="object")


def witness_birch_subcluster_squared_radius(state: object) -> AbstractArray:
    """Describe the scalar squared-radius value from BIRCH subcluster statistics."""
    del state
    return AbstractArray(shape=(), dtype="float64")


def witness_birch_subcluster_merge(
    base_state: object,
    nominee_state: object,
    threshold: float,
) -> AbstractArray:
    """Describe the merge flag and resulting BIRCH subcluster statistics."""
    del base_state, nominee_state, threshold
    return AbstractArray(shape=(), dtype="object")


def witness_birch_subcluster_radius(state: object) -> AbstractArray:
    """Describe the scalar radius value from BIRCH subcluster statistics."""
    del state
    return AbstractArray(shape=(), dtype="float64")
