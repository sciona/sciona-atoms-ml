"""Ghost witnesses for MinCovDet fit-prelude atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_matrix(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    rows, cols = int(values.shape[0]), int(values.shape[1])
    if rows < 2 or cols < 1:
        raise ValueError(f"{name} must have at least two rows and one column")
    return rows, cols


def witness_mincovdet_fit_validated_data(
    X: AbstractArray,
) -> AbstractArray:
    """Describe MinCovDet's validated fit input matrix."""
    rows, cols = _check_matrix(X, "X")
    return AbstractArray(shape=(rows, cols), dtype="float64")


def witness_mincovdet_fit_random_state(
    random_state: object,
) -> AbstractArray:
    """Describe MinCovDet's normalized random-state object."""
    del random_state
    return AbstractArray(shape=(), dtype="object")


def witness_mincovdet_fit_shape(
    X: AbstractArray,
) -> AbstractArray:
    """Describe the `(n_samples, n_features)` tuple unpacked in MinCovDet.fit."""
    rows, cols = _check_matrix(X, "X")
    del rows, cols
    return AbstractArray(shape=(2,), dtype="int64")


def witness_mincovdet_fit_assume_centered_branch(
    assume_centered: bool,
) -> bool:
    """Describe the assume-centered branch predicate in MinCovDet.fit."""
    return bool(assume_centered)
