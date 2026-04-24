"""Ghost witnesses for Gaussian-process regression preprocessing helper atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_vector_or_matrix(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) == 1:
        rows = int(values.shape[0])
        if rows < 1:
            raise ValueError(f"{name} must be nonempty")
        return rows, 1
    if len(values.shape) == 2:
        rows, cols = int(values.shape[0]), int(values.shape[1])
        if rows < 1 or cols < 1:
            raise ValueError(f"{name} must be nonempty")
        return rows, cols
    raise ValueError(f"{name} must be a vector or matrix")


def witness_gp_regression_target_count(y: AbstractArray) -> int:
    """Describe the observed target count in a GP regression fit input."""
    _, cols = _check_vector_or_matrix(y, "y")
    return cols


def witness_gp_regression_validate_n_targets(
    observed_n_targets: int,
    *,
    n_targets: int | None = None,
) -> int:
    """Describe validating the observed target count against a configured one."""
    if observed_n_targets < 1:
        raise ValueError("observed_n_targets must be positive")
    if n_targets is not None and n_targets < 1:
        raise ValueError("n_targets must be positive when provided")
    return observed_n_targets


def witness_gp_regression_target_statistics(
    y: AbstractArray,
    *,
    normalize_y: bool = False,
) -> tuple[float | AbstractArray, float | AbstractArray]:
    """Describe GP regression target mean and scale statistics."""
    del normalize_y
    _, cols = _check_vector_or_matrix(y, "y")
    if len(y.shape) == 1:
        return 0.0, 1.0
    return AbstractArray(shape=(cols,), dtype="float64"), AbstractArray(shape=(cols,), dtype="float64")


def witness_gp_regression_scaled_targets(
    y: AbstractArray,
    y_train_mean: float | AbstractArray,
    y_train_std: float | AbstractArray,
) -> AbstractArray:
    """Describe target values after GP regression scaling."""
    del y_train_mean, y_train_std
    rows, cols = _check_vector_or_matrix(y, "y")
    if len(y.shape) == 1:
        return AbstractArray(shape=(rows,), dtype="float64")
    return AbstractArray(shape=(rows, cols), dtype="float64")


def witness_gp_regression_resolve_alpha(
    alpha: float | AbstractArray,
    *,
    n_samples: int,
) -> float | AbstractArray:
    """Describe resolved GP regression alpha noise values."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if isinstance(alpha, (int, float)) and not isinstance(alpha, bool):
        return 0.0
    if len(alpha.shape) != 1:
        raise ValueError("alpha must be a vector when not scalar")
    if int(alpha.shape[0]) == 1:
        return 0.0
    return AbstractArray(shape=(n_samples,), dtype="float64")
