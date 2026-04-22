"""Ghost witnesses for sklearn Huber-regression objective helpers."""

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


def witness_huber_linear_residuals(
    X: AbstractArray,
    y: AbstractArray,
    coef: AbstractArray,
    *,
    intercept: float = 0.0,
) -> AbstractArray:
    """Describe residuals from a supplied linear prediction."""
    del intercept
    n_samples, n_features = _check_matrix(X, "X")
    if _check_vector(y, "y") != n_samples:
        raise ValueError("y must match X samples")
    if _check_vector(coef, "coef") != n_features:
        raise ValueError("coef must match X features")
    return AbstractArray(shape=(n_samples,), dtype="float64")


def witness_huber_outlier_mask(
    residuals: AbstractArray,
    *,
    epsilon: float,
    sigma: float,
) -> AbstractArray:
    """Describe which residuals are outside the Huber quadratic region."""
    del epsilon, sigma
    n_samples = _check_vector(residuals, "residuals")
    return AbstractArray(shape=(n_samples,), dtype="bool")


def witness_huber_loss_gradient(
    params: AbstractArray,
    X: AbstractArray,
    y: AbstractArray,
    *,
    epsilon: float,
    alpha: float,
    sample_weight: AbstractArray | None = None,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe the Huber objective value and parameter gradient."""
    del epsilon, alpha
    n_params = _check_vector(params, "params")
    n_samples, n_features = _check_matrix(X, "X")
    if _check_vector(y, "y") != n_samples:
        raise ValueError("y must match X samples")
    if n_params not in {n_features + 1, n_features + 2}:
        raise ValueError("params must contain coef, optional intercept, and scale")
    if sample_weight is not None and _check_vector(sample_weight, "sample_weight") != n_samples:
        raise ValueError("sample_weight must match X samples")
    return (
        AbstractArray(shape=(), dtype="float64"),
        AbstractArray(shape=(n_params,), dtype="float64"),
    )
