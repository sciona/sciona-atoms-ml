"""Ghost witnesses for selected sklearn dummy estimator atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray

from .state_models import DummyRegressorState


def witness_dummy_regressor_fit(
    y: AbstractArray,
    *,
    strategy: str = "mean",
    constant: float | tuple[float, ...] | None = None,
    quantile: float | None = None,
) -> AbstractArray:
    """Describe learning the constant value emitted by a dummy regressor."""
    del constant
    if len(y.shape) not in {1, 2}:
        raise ValueError("y must be 1D or 2D")
    if int(y.shape[0]) < 1:
        raise ValueError("y must contain at least one sample")
    if strategy not in {"mean", "median", "quantile", "constant"}:
        raise ValueError("invalid dummy regressor strategy")
    if strategy == "quantile" and (quantile is None or not 0.0 <= quantile <= 1.0):
        raise ValueError("quantile must lie in [0, 1]")
    n_outputs = 1 if len(y.shape) == 1 else int(y.shape[1])
    return AbstractArray(shape=(1, n_outputs), dtype="float64")


def witness_dummy_regressor_predict(
    X: AbstractArray,
    state: DummyRegressorState,
) -> AbstractArray:
    """Describe broadcasting the fitted dummy-regressor constant."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    n_samples = int(X.shape[0])
    if state.n_outputs == 1:
        return AbstractArray(shape=(n_samples,), dtype="float64")
    return AbstractArray(shape=(n_samples, state.n_outputs), dtype="float64")
