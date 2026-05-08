"""Selected dummy estimator atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .state_models import DummyClassifierState, DummyRegressorState
from .witnesses import (
    witness_dummy_classifier_fit,
    witness_dummy_classifier_predict,
    witness_dummy_classifier_predict_proba,
    witness_dummy_regressor_fit,
    witness_dummy_regressor_predict,
)

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

def _classifier_strategy_valid(strategy: str) -> bool:
    return strategy in {"prior", "most_frequent", "constant"}

def _finite_target(y: NDArray[np.float64]) -> bool:
    values = np.asarray(y, dtype=np.float64)
    return bool(values.ndim in {1, 2} and values.size > 0 and np.all(np.isfinite(values)))

def _classifier_constant_valid(strategy: str, constant: float | tuple[float, ...] | None) -> bool:
    return strategy != "constant" or constant is not None

def _classifier_state_valid(state: DummyClassifierState) -> bool:
    return bool(
        state.n_outputs >= 1
        and len(state.classes) == state.n_outputs
        and len(state.class_prior) == state.n_outputs
        and len(state.n_classes) == state.n_outputs
        and state.strategy in {"prior", "most_frequent", "constant"}
        and (state.constant is None or len(state.constant) == state.n_outputs)
        and all(classes.ndim == 1 and classes.shape[0] == n_classes for classes, n_classes in zip(state.classes, state.n_classes))
        and all(prior.ndim == 1 and prior.shape[0] == n_classes for prior, n_classes in zip(state.class_prior, state.n_classes))
        and all(np.all(np.isfinite(classes)) for classes in state.classes)
        and all(np.all(np.isfinite(prior)) and np.all(prior >= 0.0) and np.isclose(np.sum(prior), 1.0) for prior in state.class_prior)
        and (state.constant is None or all(np.isfinite(value) for value in state.constant))
    )

def _classifier_prediction_valid(result: NDArray[np.float64], state: DummyClassifierState) -> bool:
    values = np.asarray(result)
    expected_ndim = 1 if state.n_outputs == 1 else 2
    return bool(
        values.ndim == expected_ndim
        and (state.n_outputs == 1 or values.shape[1] == state.n_outputs)
        and np.all(np.isfinite(values))
    )

def _classifier_proba_valid(result: NDArray[np.float64] | tuple[NDArray[np.float64], ...], state: DummyClassifierState) -> bool:
    arrays = (result,) if isinstance(result, np.ndarray) else result
    return bool(
        len(arrays) == state.n_outputs
        and all(array.ndim == 2 for array in arrays)
        and all(array.shape[1] == state.n_classes[index] for index, array in enumerate(arrays))
        and all(np.all(np.isfinite(array)) and np.all(array >= 0.0) and np.allclose(array.sum(axis=1), 1.0) for array in arrays)
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
    from sklearn.utils import check_array
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
    from sklearn.utils import check_array
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

@register_atom(witness_dummy_classifier_fit)
@icontract.require(lambda y: _target_1d_or_2d(y), "y must be 1D or 2D")
@icontract.require(lambda y: _nonempty_target(y), "y must contain at least one target value")
@icontract.require(lambda y: _finite_target(y), "y labels must be finite numeric values")
@icontract.require(lambda strategy: _classifier_strategy_valid(strategy), "strategy must be prior, most_frequent, or constant")
@icontract.require(lambda strategy, constant: _classifier_constant_valid(strategy, constant), "constant strategy requires a constant")
@icontract.ensure(lambda result: _classifier_state_valid(result), "dummy classifier state must contain finite class priors")
def dummy_classifier_fit(
    y: NDArray[np.float64],
    *,
    strategy: str = "prior",
    constant: float | tuple[float, ...] | None = None,
) -> DummyClassifierState:
    from sklearn.utils import check_array
    """Fit deterministic class priors used by sklearn's dummy classifier."""
    checked_y = check_array(y, dtype=np.float64, ensure_2d=False, input_name="y")
    if checked_y.ndim == 1:
        checked_y = np.reshape(checked_y, (-1, 1))
    n_outputs = int(checked_y.shape[1])

    if strategy == "constant":
        constant_values = tuple(float(value) for value in np.asarray(constant, dtype=np.float64).reshape(-1))
        if len(constant_values) != n_outputs:
            raise ValueError(f"Constant target value should have shape ({n_outputs}, 1).")
    else:
        constant_values = None

    classes: list[NDArray[np.float64]] = []
    class_prior: list[NDArray[np.float64]] = []
    n_classes: list[int] = []
    for output_index in range(n_outputs):
        classes_k, encoded = np.unique(checked_y[:, output_index], return_inverse=True)
        prior_k = np.bincount(encoded).astype(np.float64)
        prior_k /= np.sum(prior_k)
        if constant_values is not None and constant_values[output_index] not in classes_k:
            raise ValueError("The constant target value must be present in the training data.")
        classes.append(np.asarray(classes_k, dtype=np.float64).copy())
        class_prior.append(np.asarray(prior_k, dtype=np.float64).copy())
        n_classes.append(int(classes_k.shape[0]))

    return DummyClassifierState(
        classes=tuple(classes),
        class_prior=tuple(class_prior),
        n_classes=tuple(n_classes),
        n_outputs=n_outputs,
        strategy=strategy,
        constant=constant_values,
    )

@register_atom(witness_dummy_classifier_predict)
@icontract.require(lambda X: np.asarray(X).ndim == 2, "X must be 2D")
@icontract.require(lambda state: _classifier_state_valid(state), "state must be a fitted dummy classifier state")
@icontract.ensure(lambda result, state: _classifier_prediction_valid(result, state), "predictions must broadcast deterministic dummy labels")
def dummy_classifier_predict(
    X: NDArray[np.float64],
    state: DummyClassifierState,
) -> NDArray[np.float64]:
    from sklearn.utils import check_array
    """Predict deterministic dummy-classifier labels for each row."""
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    per_output: list[float] = []
    for output_index in range(state.n_outputs):
        if state.strategy in {"prior", "most_frequent"}:
            label = float(state.classes[output_index][np.argmax(state.class_prior[output_index])])
        else:
            if state.constant is None:
                raise ValueError("constant strategy requires fitted constants")
            label = float(state.constant[output_index])
        per_output.append(label)
    predictions = np.tile(np.asarray(per_output, dtype=np.float64), (checked_x.shape[0], 1))
    if state.n_outputs == 1:
        return np.ravel(predictions).astype(np.float64)
    return np.asarray(predictions, dtype=np.float64)

@register_atom(witness_dummy_classifier_predict_proba)
@icontract.require(lambda X: np.asarray(X).ndim == 2, "X must be 2D")
@icontract.require(lambda state: _classifier_state_valid(state), "state must be a fitted dummy classifier state")
@icontract.ensure(lambda result, state: _classifier_proba_valid(result, state), "probabilities must be valid per-class rows")
def dummy_classifier_predict_proba(
    X: NDArray[np.float64],
    state: DummyClassifierState,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], ...]:
    from sklearn.utils import check_array
    """Predict deterministic dummy-classifier class probabilities for each row."""
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    outputs: list[NDArray[np.float64]] = []
    for output_index in range(state.n_outputs):
        if state.strategy == "prior":
            out = np.ones((checked_x.shape[0], 1), dtype=np.float64) * state.class_prior[output_index]
        elif state.strategy == "most_frequent":
            out = np.zeros((checked_x.shape[0], state.n_classes[output_index]), dtype=np.float64)
            out[:, int(np.argmax(state.class_prior[output_index]))] = 1.0
        else:
            if state.constant is None:
                raise ValueError("constant strategy requires fitted constants")
            out = np.zeros((checked_x.shape[0], state.n_classes[output_index]), dtype=np.float64)
            constant_index = np.where(state.classes[output_index] == state.constant[output_index])[0]
            out[:, constant_index] = 1.0
        outputs.append(out)
    if state.n_outputs == 1:
        return outputs[0]
    return tuple(outputs)
