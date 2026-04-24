"""Ghost witnesses for Gaussian-process regression sampling helper atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_vector(values: AbstractArray, name: str) -> int:
    if len(values.shape) != 1:
        raise ValueError(f"{name} must be 1D")
    size = int(values.shape[0])
    if size < 1:
        raise ValueError(f"{name} must be nonempty")
    return size


def _check_matrix(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    rows, cols = int(values.shape[0]), int(values.shape[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"{name} must be nonempty")
    return rows, cols


def witness_gp_sample_y_single_output(
    y_mean: AbstractArray,
    y_cov: AbstractArray,
    *,
    n_samples: int = 1,
    random_state: int | None = 0,
) -> AbstractArray:
    """Describe single-output GP sample draws from a predictive Gaussian."""
    del random_state
    n_points = _check_vector(y_mean, "y_mean")
    rows, cols = _check_matrix(y_cov, "y_cov")
    if rows != n_points or cols != n_points:
        raise ValueError("y_cov must be square with one row and column per point")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    return AbstractArray(shape=(n_points, n_samples), dtype="float64")


def witness_gp_sample_y_multi_output(
    y_mean: AbstractArray,
    y_cov: AbstractArray,
    *,
    n_samples: int = 1,
    random_state: int | None = 0,
) -> AbstractArray:
    """Describe multi-output GP sample draws from per-target predictive Gaussians."""
    del random_state
    n_points, n_targets = _check_matrix(y_mean, "y_mean")
    if len(y_cov.shape) != 3:
        raise ValueError("y_cov must be 3D")
    if int(y_cov.shape[0]) != n_points or int(y_cov.shape[1]) != n_points or int(y_cov.shape[2]) != n_targets:
        raise ValueError("y_cov must have shape (n_points, n_points, n_targets)")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    return AbstractArray(shape=(n_points, n_targets, n_samples), dtype="float64")
