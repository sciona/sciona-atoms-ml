"""Ghost witnesses for sklearn quantile-regression LP helpers."""

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


def witness_quantile_nonzero_weight_mask(sample_weight: AbstractArray) -> AbstractArray:
    """Describe which sample weights participate in the LP system."""
    n_samples = _check_vector(sample_weight, "sample_weight")
    return AbstractArray(shape=(n_samples,), dtype="bool")


def witness_quantile_dense_lp_problem(
    X: AbstractArray,
    y: AbstractArray,
    sample_weight: AbstractArray,
    *,
    quantile: float,
    alpha: float,
    fit_intercept: bool = True,
) -> tuple[AbstractArray, AbstractArray, AbstractArray]:
    """Describe dense quantile-regression LP arrays."""
    del quantile, alpha
    n_samples, n_features = _check_matrix(X, "X")
    if _check_vector(y, "y") != n_samples:
        raise ValueError("y must match X samples")
    if _check_vector(sample_weight, "sample_weight") != n_samples:
        raise ValueError("sample_weight must match X samples")
    n_params = n_features + int(fit_intercept)
    n_columns = 2 * n_params + 2 * n_samples
    return (
        AbstractArray(shape=(n_columns,), dtype="float64"),
        AbstractArray(shape=(n_samples, n_columns), dtype="float64"),
        AbstractArray(shape=(n_samples,), dtype="float64"),
    )


def witness_quantile_solution_to_params(
    solution: AbstractArray,
    n_features: int,
    *,
    fit_intercept: bool = True,
) -> tuple[AbstractArray, float]:
    """Describe converting LP parameter slacks to model coefficients."""
    n_solution = _check_vector(solution, "solution")
    n_params = int(n_features) + int(fit_intercept)
    if n_features < 1:
        raise ValueError("n_features must be positive")
    if n_solution < 2 * n_params:
        raise ValueError("solution must contain positive and negative parameter slacks")
    return AbstractArray(shape=(int(n_features),), dtype="float64"), 0.0
