"""Selected dummy estimator atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from sklearn.utils import check_array

from sciona.ghost.registry import register_atom

from .state_models import DummyRegressorState
from .witnesses import witness_dummy_regressor_fit, witness_dummy_regressor_predict


def _target_1d_or_2d(y: NDArray[np.float64]) -> bool:
    return bool(np.asarray(y).ndim in {1, 2})


def _nonempty_target(y: NDArray[np.float64]) -> bool:
    values = np.asarray(y)
    return bool(values.ndim in {1, 2} and values.shape[0] > 0 and values.size > 0)


def _strategy_valid(strategy: str) -> bool:
    return strategy in {"mean", "median", "quantile", "constant"}


def _quantile_valid(strategy: str, quantile: float | None) -> bool:
    if strategy != "quantile":
        return True
    return quantile is not None and 0.0 <= quantile <= 1.0


def _constant_valid(strategy: str, constant: float | tuple[float, ...] | None) -> bool:
    return strategy != "constant" or constant is not None


def _state_valid(state: DummyRegressorState) -> bool:
    return bool(
        state.constant.shape == (1, state.n_outputs)
        and state.n_outputs >= 1
        and state.strategy in {"mean", "median", "quantile", "constant"}
        and np.all(np.isfinite(state.constant))
        and (state.quantile is None or 0.0 <= state.quantile <= 1.0)
    )


def _prediction_valid(result: NDArray[np.float64], state: DummyRegressorState) -> bool:
    values = np.asarray(result)
    expected_ndim = 1 if state.n_outputs == 1 else 2
    return bool(
        values.ndim == expected_ndim
        and (state.n_outputs == 1 or values.shape[1] == state.n_outputs)
        and np.all(np.isfinite(values))
    )


@register_atom(witness_dummy_regressor_fit)
@icontract.require(lambda y: _target_1d_or_2d(y), "y must be 1D or 2D")
@icontract.require(lambda y: _nonempty_target(y), "y must contain at least one target value")
@icontract.require(lambda strategy: _strategy_valid(strategy), "strategy must be mean, median, quantile, or constant")
@icontract.require(lambda strategy, quantile: _quantile_valid(strategy, quantile), "quantile strategy requires a quantile in [0, 1]")
@icontract.require(lambda strategy, constant: _constant_valid(strategy, constant), "constant strategy requires a constant")
@icontract.ensure(lambda result: _state_valid(result), "dummy regressor state must contain a finite prediction constant")
def dummy_regressor_fit(
    y: NDArray[np.float64],
    *,
    strategy: str = "mean",
    constant: float | tuple[float, ...] | None = None,
    quantile: float | None = None,
) -> DummyRegressorState:
    """Fit the constant target value used by sklearn's dummy regressor."""
    checked_y = check_array(y, dtype=np.float64, ensure_2d=False, input_name="y")
    if checked_y.ndim == 1:
        checked_y = np.reshape(checked_y, (-1, 1))
    n_outputs = int(checked_y.shape[1])

    if strategy == "mean":
        fitted_constant = np.mean(checked_y, axis=0)
        fitted_quantile = None
    elif strategy == "median":
        fitted_constant = np.median(checked_y, axis=0)
        fitted_quantile = None
    elif strategy == "quantile":
        if quantile is None:
            raise ValueError("quantile strategy requires a quantile")
        fitted_constant = np.percentile(checked_y, axis=0, q=quantile * 100.0)
        fitted_quantile = float(quantile)
    else:
        constant_values = np.asarray(constant, dtype=np.float64).reshape(-1)
        if n_outputs != 1 and constant_values.shape[0] != n_outputs:
            raise ValueError(f"Constant target value should have shape ({n_outputs}, 1).")
        fitted_constant = constant_values
        fitted_quantile = None

    return DummyRegressorState(
        constant=np.asarray(fitted_constant, dtype=np.float64).reshape(1, -1).copy(),
        n_outputs=n_outputs,
        strategy=strategy,
        quantile=fitted_quantile,
    )


@register_atom(witness_dummy_regressor_predict)
@icontract.require(lambda X: np.asarray(X).ndim == 2, "X must be 2D")
@icontract.require(lambda state: _state_valid(state), "state must be a fitted dummy regressor state")
@icontract.ensure(lambda result, state: _prediction_valid(result, state), "predictions must broadcast the fitted constant")
def dummy_regressor_predict(
    X: NDArray[np.float64],
    state: DummyRegressorState,
) -> NDArray[np.float64]:
    """Predict by repeating the fitted dummy-regressor constant for each row."""
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    predictions = np.full(
        (checked_x.shape[0], state.n_outputs),
        state.constant,
        dtype=np.asarray(state.constant).dtype,
    )
    if state.n_outputs == 1:
        return np.ravel(predictions).astype(np.float64)
    return np.asarray(predictions, dtype=np.float64)
