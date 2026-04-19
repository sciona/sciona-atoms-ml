"""Ghost witnesses for selected sklearn SVM helper atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_l1_min_c(
    X: AbstractArray,
    y: AbstractArray,
    *,
    loss: str = "squared_hinge",
    fit_intercept: bool = True,
    intercept_scaling: float = 1.0,
) -> AbstractArray:
    """Describe a positive scalar C lower bound for l1-penalized classifiers."""
    del fit_intercept
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if len(y.shape) != 1:
        raise ValueError("y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have equal sample count")
    if loss not in {"squared_hinge", "log"}:
        raise ValueError("loss must be 'squared_hinge' or 'log'")
    if intercept_scaling <= 0.0:
        raise ValueError("intercept_scaling must be positive")
    return AbstractArray(shape=(), dtype="float64", min_val=0.0)
