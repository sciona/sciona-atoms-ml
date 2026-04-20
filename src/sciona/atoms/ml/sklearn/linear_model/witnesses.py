"""Ghost witnesses for sklearn linear model atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray

from .state_models import LinearRegressionState


def witness_linear_regression_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    fit_intercept: bool = True,
    copy_X: bool = True,
    tol: float = 1e-6,
    n_jobs: int | None = None,
    positive: bool = False,
    sample_weight: float | tuple[float, ...] | None = None,
) -> AbstractArray:
    """Describe fitting dense ordinary least-squares coefficients."""
    del copy_X, tol, n_jobs, sample_weight
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if len(y.shape) not in {1, 2}:
        raise ValueError("y must be 1D or 2D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have matching sample counts")
    if not isinstance(fit_intercept, bool):
        raise ValueError("fit_intercept must be boolean")
    if positive:
        raise ValueError("positive=True is outside the dense OLS atom scope")
    n_outputs = 1 if len(y.shape) == 1 else int(y.shape[1])
    if n_outputs == 1:
        return AbstractArray(shape=(int(X.shape[1]),), dtype="float64")
    return AbstractArray(shape=(n_outputs, int(X.shape[1])), dtype="float64")


def witness_linear_regression_predict(X: AbstractArray, state: LinearRegressionState) -> AbstractArray:
    """Describe predicting with fitted ordinary least-squares coefficients."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    if state.n_outputs == 1:
        return AbstractArray(shape=(int(X.shape[0]),), dtype="float64")
    return AbstractArray(shape=(int(X.shape[0]), state.n_outputs), dtype="float64")
