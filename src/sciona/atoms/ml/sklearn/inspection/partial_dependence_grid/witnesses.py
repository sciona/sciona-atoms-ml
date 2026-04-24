"""Ghost witnesses for partial-dependence grid helper atoms."""

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


def witness_partial_dependence_grid_parameters(
    percentiles: tuple[float, float],
    *,
    grid_resolution: int,
) -> tuple[float, float]:
    """Describe validated partial-dependence grid parameters."""
    del percentiles, grid_resolution
    return 0.0, 1.0


def witness_partial_dependence_feature_axis(
    feature_values: AbstractArray,
    *,
    percentiles: tuple[float, float],
    is_categorical: bool,
    grid_resolution: int,
) -> AbstractArray:
    """Describe one feature axis used in a partial-dependence grid."""
    del percentiles, is_categorical, grid_resolution
    _check_vector(feature_values, "feature_values")
    return AbstractArray(shape=(1,), dtype="float64")


def witness_partial_dependence_grid(
    X: AbstractArray,
    *,
    percentiles: tuple[float, float],
    is_categorical: tuple[bool, ...],
    grid_resolution: int,
) -> tuple[AbstractArray, tuple[AbstractArray, ...]]:
    """Describe the Cartesian grid and per-feature axes."""
    _, n_features = _check_matrix(X, "X")
    if len(is_categorical) != n_features:
        raise ValueError("is_categorical must match the number of features")
    del percentiles, grid_resolution
    return (
        AbstractArray(shape=(1, n_features), dtype="float64"),
        tuple(AbstractArray(shape=(1,), dtype="float64") for _ in range(n_features)),
    )
