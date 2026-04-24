"""Ghost witnesses for Gaussian-process regression optimizer bookkeeping atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_bounds_matrix(bounds: AbstractArray, name: str) -> int:
    if len(bounds.shape) != 2:
        raise ValueError(f"{name} must be a matrix")
    rows, cols = int(bounds.shape[0]), int(bounds.shape[1])
    if rows < 1:
        raise ValueError(f"{name} must be nonempty")
    if cols != 2:
        raise ValueError(f"{name} must have exactly two columns")
    return rows


def _check_vector(values: AbstractArray, name: str) -> int:
    if len(values.shape) != 1:
        raise ValueError(f"{name} must be a vector")
    size = int(values.shape[0])
    if size < 1:
        raise ValueError(f"{name} must be nonempty")
    return size


def witness_gp_regression_restart_bounds(
    bounds: AbstractArray,
    *,
    n_restarts_optimizer: int = 0,
) -> AbstractArray:
    """Describe checking whether restart bounds can be used."""
    if n_restarts_optimizer < 0:
        raise ValueError("n_restarts_optimizer must be nonnegative")
    rows = _check_bounds_matrix(bounds, "bounds")
    return AbstractArray(shape=(rows, 2), dtype="float64")


def witness_gp_regression_restart_thetas(
    bounds: AbstractArray,
    *,
    n_restarts_optimizer: int,
    random_state: int | None = 0,
) -> AbstractArray:
    """Describe drawing starting values for repeated searches."""
    del random_state
    if n_restarts_optimizer < 0:
        raise ValueError("n_restarts_optimizer must be nonnegative")
    rows = _check_bounds_matrix(bounds, "bounds")
    return AbstractArray(shape=(n_restarts_optimizer, rows), dtype="float64")


def witness_gp_regression_select_best_optimum(
    candidate_thetas: AbstractArray,
    objective_values: AbstractArray,
) -> tuple[AbstractArray, float]:
    """Describe selecting the best optimizer result from candidate objective values."""
    if len(candidate_thetas.shape) != 2:
        raise ValueError("candidate_thetas must be a matrix")
    rows, cols = int(candidate_thetas.shape[0]), int(candidate_thetas.shape[1])
    if rows < 1 or cols < 1:
        raise ValueError("candidate_thetas must be nonempty")
    objective_rows = _check_vector(objective_values, "objective_values")
    if objective_rows != rows:
        raise ValueError("objective_values must align with candidate_thetas rows")
    return AbstractArray(shape=(cols,), dtype="float64"), 0.0
