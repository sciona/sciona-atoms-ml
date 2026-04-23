"""Dense MLP helper atoms adapted from scikit-learn."""

from __future__ import annotations

from typing import Literal

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, xlogy

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_mlp_activation,
    witness_mlp_activation_derivative,
    witness_mlp_backprop,
    witness_mlp_forward_pass,
    witness_mlp_layer_gradients,
    witness_mlp_loss,
)

ActivationName = Literal["identity", "logistic", "tanh", "relu", "softmax"]
HiddenActivationName = Literal["identity", "logistic", "tanh", "relu"]
LossName = Literal["log_loss", "squared_error"]
WeightTuple = tuple[NDArray[np.float64], ...]
BiasTuple = tuple[NDArray[np.float64], ...]
ActivationTuple = tuple[NDArray[np.float64], ...]
GradientTuple = tuple[NDArray[np.float64], ...]
LayerGradient = tuple[NDArray[np.float64], NDArray[np.float64]]
BackpropResult = tuple[float, GradientTuple, GradientTuple]

_ACTIVATIONS = {"identity", "logistic", "tanh", "relu", "softmax"}
_HIDDEN_ACTIVATIONS = {"identity", "logistic", "tanh", "relu"}
_LOSS_NAMES = {"log_loss", "squared_error"}


def _finite_matrix(values: NDArray[np.float64]) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _activation_name_valid(activation: str) -> bool:
    return bool(isinstance(activation, str) and activation in _ACTIVATIONS)


def _hidden_activation_name_valid(activation: str) -> bool:
    return bool(isinstance(activation, str) and activation in _HIDDEN_ACTIVATIONS)


def _loss_name_valid(loss_name: str) -> bool:
    return bool(isinstance(loss_name, str) and loss_name in _LOSS_NAMES)


def _canonical_loss_combo_valid(loss_name: str, output_activation: str) -> bool:
    return bool(
        _loss_name_valid(loss_name)
        and _activation_name_valid(output_activation)
        and (
            (loss_name == "squared_error" and output_activation == "identity")
            or (loss_name == "log_loss" and output_activation in {"logistic", "softmax"})
        )
    )


def _nonnegative_scalar(value: float) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)) and float(value) >= 0.0)


def _sample_weight_valid(sample_weight: NDArray[np.float64] | None, n_samples: int) -> bool:
    if sample_weight is None:
        return True
    try:
        values = np.asarray(sample_weight, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        values.ndim == 1
        and values.shape[0] == n_samples
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and float(np.sum(values)) > 0.0
    )


def _weight_tuple_valid(coefs: WeightTuple) -> bool:
    if not isinstance(coefs, tuple) or len(coefs) < 1:
        return False
    previous_out: int | None = None
    for coef in coefs:
        try:
            values = np.asarray(coef, dtype=np.float64)
        except (TypeError, ValueError):
            return False
        if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1 or not np.all(np.isfinite(values)):
            return False
        if previous_out is not None and values.shape[0] != previous_out:
            return False
        previous_out = int(values.shape[1])
    return True


def _bias_tuple_valid(intercepts: BiasTuple, coefs: WeightTuple | None = None) -> bool:
    if not isinstance(intercepts, tuple) or len(intercepts) < 1:
        return False
    if coefs is not None and len(intercepts) != len(coefs):
        return False
    for index, intercept in enumerate(intercepts):
        try:
            values = np.asarray(intercept, dtype=np.float64)
        except (TypeError, ValueError):
            return False
        if values.ndim != 1 or values.shape[0] < 1 or not np.all(np.isfinite(values)):
            return False
        if coefs is not None and values.shape[0] != np.asarray(coefs[index], dtype=np.float64).shape[1]:
            return False
    return True


def _network_inputs_valid(X: NDArray[np.float64], coefs: WeightTuple, intercepts: BiasTuple) -> bool:
    samples = np.asarray(X, dtype=np.float64)
    return bool(
        _finite_matrix(X)
        and _weight_tuple_valid(coefs)
        and _bias_tuple_valid(intercepts, coefs)
        and samples.shape[1] == np.asarray(coefs[0], dtype=np.float64).shape[0]
    )


def _targets_valid(y: NDArray[np.float64], X: NDArray[np.float64], coefs: WeightTuple) -> bool:
    targets = np.asarray(y, dtype=np.float64)
    samples = np.asarray(X, dtype=np.float64)
    return bool(
        targets.ndim == 2
        and targets.shape[0] == samples.shape[0]
        and targets.shape[1] == np.asarray(coefs[-1], dtype=np.float64).shape[1]
        and np.all(np.isfinite(targets))
    )


def _log_targets_valid(y_true: NDArray[np.float64], y_pred: NDArray[np.float64], output_activation: str) -> bool:
    targets = np.asarray(y_true, dtype=np.float64)
    predictions = np.asarray(y_pred, dtype=np.float64)
    if targets.ndim != 2 or predictions.ndim != 2 or targets.shape != predictions.shape:
        return False
    if not (
        np.all(np.isfinite(targets))
        and np.all(np.isfinite(predictions))
        and np.all(targets >= 0.0)
        and np.all(targets <= 1.0)
        and np.all(predictions >= 0.0)
        and np.all(predictions <= 1.0)
    ):
        return False
    if output_activation == "softmax":
        return bool(predictions.shape[1] >= 2 and np.allclose(predictions.sum(axis=1), 1.0))
    return True


def _sample_weight_sum_valid(sample_weight_sum: float | None) -> bool:
    return bool(sample_weight_sum is None or _nonnegative_scalar(sample_weight_sum))


def _activation_result_valid(result: NDArray[np.float64], values: NDArray[np.float64], activation: str) -> bool:
    transformed = np.asarray(result, dtype=np.float64)
    original = np.asarray(values, dtype=np.float64)
    if transformed.shape != original.shape or not np.all(np.isfinite(transformed)):
        return False
    if activation == "identity":
        return bool(np.array_equal(transformed, original))
    if activation == "logistic":
        return bool(np.all(transformed >= 0.0) and np.all(transformed <= 1.0))
    if activation == "tanh":
        return bool(np.all(transformed >= -1.0) and np.all(transformed <= 1.0))
    if activation == "relu":
        return bool(np.all(transformed >= 0.0))
    return bool(np.all(transformed >= 0.0) and np.allclose(transformed.sum(axis=1), 1.0))


def _activation_derivative_inputs_valid(
    activated_values: NDArray[np.float64],
    delta: NDArray[np.float64],
    activation: str,
) -> bool:
    activated = np.asarray(activated_values, dtype=np.float64)
    delta_values = np.asarray(delta, dtype=np.float64)
    return bool(
        _hidden_activation_name_valid(activation)
        and _finite_matrix(activated_values)
        and _finite_matrix(delta)
        and activated.shape == delta_values.shape
        and _activation_result_valid(activated, activated, activation)
    )


def _forward_pass_result_valid(result: ActivationTuple, X: NDArray[np.float64], coefs: WeightTuple) -> bool:
    samples = np.asarray(X, dtype=np.float64)
    if not isinstance(result, tuple) or len(result) != len(coefs) + 1:
        return False
    if np.asarray(result[0], dtype=np.float64).shape != samples.shape:
        return False
    for index, activation in enumerate(result):
        values = np.asarray(activation, dtype=np.float64)
        if values.ndim != 2 or not np.all(np.isfinite(values)):
            return False
        if index == 0:
            continue
        expected_width = np.asarray(coefs[index - 1], dtype=np.float64).shape[1]
        if values.shape != (samples.shape[0], expected_width):
            return False
    return True


def _layer_gradient_inputs_valid(
    layer_activation: NDArray[np.float64],
    delta: NDArray[np.float64],
    coefs: NDArray[np.float64],
) -> bool:
    activation_values = np.asarray(layer_activation, dtype=np.float64)
    delta_values = np.asarray(delta, dtype=np.float64)
    coef_values = np.asarray(coefs, dtype=np.float64)
    return bool(
        _finite_matrix(layer_activation)
        and _finite_matrix(delta)
        and _finite_matrix(coefs)
        and activation_values.shape[0] == delta_values.shape[0]
        and activation_values.shape[1] == coef_values.shape[0]
        and delta_values.shape[1] == coef_values.shape[1]
    )


def _layer_gradient_result_valid(result: LayerGradient, coefs: NDArray[np.float64], delta: NDArray[np.float64]) -> bool:
    coef_grad, intercept_grad = result
    coef_values = np.asarray(coefs, dtype=np.float64)
    delta_values = np.asarray(delta, dtype=np.float64)
    intercept_values = np.asarray(intercept_grad, dtype=np.float64)
    return bool(
        np.asarray(coef_grad, dtype=np.float64).shape == coef_values.shape
        and intercept_values.shape == (delta_values.shape[1],)
        and np.all(np.isfinite(np.asarray(coef_grad, dtype=np.float64)))
        and np.all(np.isfinite(intercept_values))
    )


def _backprop_result_valid(result: BackpropResult, coefs: WeightTuple, intercepts: BiasTuple) -> bool:
    loss, coef_grads, intercept_grads = result
    return bool(
        isinstance(loss, float)
        and np.isfinite(loss)
        and isinstance(coef_grads, tuple)
        and isinstance(intercept_grads, tuple)
        and len(coef_grads) == len(coefs)
        and len(intercept_grads) == len(intercepts)
        and all(
            np.asarray(grad, dtype=np.float64).shape == np.asarray(coef, dtype=np.float64).shape and np.all(np.isfinite(np.asarray(grad, dtype=np.float64)))
            for grad, coef in zip(coef_grads, coefs)
        )
        and all(
            np.asarray(grad, dtype=np.float64).shape == np.asarray(intercept, dtype=np.float64).shape and np.all(np.isfinite(np.asarray(grad, dtype=np.float64)))
            for grad, intercept in zip(intercept_grads, intercepts)
        )
    )


def _apply_activation(values: NDArray[np.float64], activation: str) -> NDArray[np.float64]:
    result = np.array(np.asarray(values, dtype=np.float64), copy=True)
    if activation == "identity":
        return result
    if activation == "logistic":
        return expit(result, out=result)
    if activation == "tanh":
        return np.tanh(result, out=result)
    if activation == "relu":
        return np.maximum(result, 0.0, out=result)
    shifted = result - result.max(axis=1, keepdims=True)
    np.exp(shifted, out=result)
    result /= result.sum(axis=1, keepdims=True)
    return result


def _apply_activation_derivative(
    activated_values: NDArray[np.float64],
    delta: NDArray[np.float64],
    activation: str,
) -> NDArray[np.float64]:
    result = np.array(np.asarray(delta, dtype=np.float64), copy=True)
    activated = np.asarray(activated_values, dtype=np.float64)
    if activation == "identity":
        return result
    if activation == "logistic":
        result *= activated
        result *= 1.0 - activated
        return result
    if activation == "tanh":
        result *= 1.0 - activated**2
        return result
    result[activated == 0.0] = 0.0
    return result


def _binary_log_loss(
    y_true: NDArray[np.float64],
    y_prob: NDArray[np.float64],
    sample_weight: NDArray[np.float64] | None = None,
) -> float:
    probabilities = np.asarray(y_prob, dtype=np.float64)
    targets = np.asarray(y_true, dtype=np.float64)
    eps = np.finfo(probabilities.dtype).eps
    probabilities = np.clip(probabilities, eps, 1.0 - eps)
    return float(-np.average(xlogy(targets, probabilities) + xlogy(1.0 - targets, 1.0 - probabilities), weights=sample_weight, axis=0).sum())


def _log_loss(
    y_true: NDArray[np.float64],
    y_prob: NDArray[np.float64],
    sample_weight: NDArray[np.float64] | None = None,
) -> float:
    probabilities = np.asarray(y_prob, dtype=np.float64)
    targets = np.asarray(y_true, dtype=np.float64)
    eps = np.finfo(probabilities.dtype).eps
    probabilities = np.clip(probabilities, eps, 1.0 - eps)
    if probabilities.shape[1] == 1:
        probabilities = np.append(1.0 - probabilities, probabilities, axis=1)
    if targets.shape[1] == 1:
        targets = np.append(1.0 - targets, targets, axis=1)
    return float(-np.average(xlogy(targets, probabilities), weights=sample_weight, axis=0).sum())


def _squared_loss(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    sample_weight: NDArray[np.float64] | None = None,
) -> float:
    targets = np.asarray(y_true, dtype=np.float64)
    predictions = np.asarray(y_pred, dtype=np.float64)
    return float(0.5 * np.average((targets - predictions) ** 2, weights=sample_weight, axis=0).mean())


@register_atom(witness_mlp_activation)
@icontract.require(lambda values: _finite_matrix(values), "values must be a finite dense matrix")
@icontract.require(lambda activation: _activation_name_valid(activation), "activation must be one of sklearn's dense MLP activations")
@icontract.ensure(lambda result, values, activation: _activation_result_valid(result, values, activation), "activation output must preserve shape and activation-specific bounds")
def mlp_activation(values: NDArray[np.float64], *, activation: ActivationName) -> NDArray[np.float64]:
    """Apply one sklearn MLP activation to a dense matrix and return the transformed values."""
    return _apply_activation(values, activation)


@register_atom(witness_mlp_activation_derivative)
@icontract.require(
    lambda activated_values, delta, activation: _activation_derivative_inputs_valid(activated_values, delta, activation),
    "activated_values and delta must be finite dense matrices with matching shape and a hidden-layer activation",
)
@icontract.ensure(lambda result, delta: np.asarray(result, dtype=np.float64).shape == np.asarray(delta, dtype=np.float64).shape, "activation derivative must preserve delta shape")
def mlp_activation_derivative(
    activated_values: NDArray[np.float64],
    delta: NDArray[np.float64],
    *,
    activation: HiddenActivationName,
) -> NDArray[np.float64]:
    """Apply one hidden-layer derivative to a dense delta matrix using already-activated values."""
    return _apply_activation_derivative(activated_values, delta, activation)


@register_atom(witness_mlp_loss)
@icontract.require(lambda y_true, y_pred: _finite_matrix(y_true) and _finite_matrix(y_pred), "y_true and y_pred must be finite dense matrices")
@icontract.require(lambda y_true, y_pred: np.asarray(y_true, dtype=np.float64).shape == np.asarray(y_pred, dtype=np.float64).shape, "y_true and y_pred must have matching shape")
@icontract.require(lambda loss_name, output_activation: _canonical_loss_combo_valid(loss_name, output_activation), "loss_name and output_activation must form one sklearn MLP canonical output-loss pair")
@icontract.require(lambda y_true, y_pred, loss_name, output_activation: loss_name != "log_loss" or _log_targets_valid(y_true, y_pred, output_activation), "log-loss targets and predictions must be bounded probabilities for the chosen output activation")
@icontract.require(lambda y_true, sample_weight: _sample_weight_valid(sample_weight, np.asarray(y_true, dtype=np.float64).shape[0]), "sample_weight must be a finite nonnegative vector over samples")
@icontract.ensure(lambda result: isinstance(result, float) and np.isfinite(result) and result >= 0.0, "loss must be a finite nonnegative scalar")
def mlp_loss(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    *,
    loss_name: LossName,
    output_activation: ActivationName,
    sample_weight: NDArray[np.float64] | None = None,
) -> float:
    """Compute one sklearn MLP output-layer loss from dense targets and predictions."""
    if loss_name == "log_loss" and output_activation == "logistic":
        return _binary_log_loss(y_true, y_pred, sample_weight)
    if loss_name == "log_loss":
        return _log_loss(y_true, y_pred, sample_weight)
    return _squared_loss(y_true, y_pred, sample_weight)


@register_atom(witness_mlp_forward_pass)
@icontract.require(lambda X, coefs, intercepts: _network_inputs_valid(X, coefs, intercepts), "X, coefs, and intercepts must define a finite dense MLP parameter chain")
@icontract.require(lambda hidden_activation: _hidden_activation_name_valid(hidden_activation), "hidden_activation must be one of sklearn's hidden-layer activations")
@icontract.require(lambda output_activation: _activation_name_valid(output_activation), "output_activation must be one of sklearn's dense MLP activations")
@icontract.require(lambda coefs, output_activation: output_activation != "softmax" or np.asarray(coefs[-1], dtype=np.float64).shape[1] >= 2, "softmax output layers must have at least two units")
@icontract.ensure(lambda result, X, coefs: _forward_pass_result_valid(result, X, coefs), "forward pass must return one finite activation matrix per layer")
def mlp_forward_pass(
    X: NDArray[np.float64],
    coefs: WeightTuple,
    intercepts: BiasTuple,
    *,
    hidden_activation: HiddenActivationName,
    output_activation: ActivationName,
) -> ActivationTuple:
    """Propagate dense samples through supplied MLP weights and biases."""
    activations: list[NDArray[np.float64]] = [np.asarray(X, dtype=np.float64)]
    current = activations[0]
    last_index = len(coefs) - 1
    for index, (coef, intercept) in enumerate(zip(coefs, intercepts)):
        current = np.asarray(current.dot(np.asarray(coef, dtype=np.float64)) + np.asarray(intercept, dtype=np.float64), dtype=np.float64)
        current = _apply_activation(current, output_activation if index == last_index else hidden_activation)
        activations.append(current)
    return tuple(np.asarray(values, dtype=np.float64) for values in activations)


@register_atom(witness_mlp_layer_gradients)
@icontract.require(lambda layer_activation, delta, coefs: _layer_gradient_inputs_valid(layer_activation, delta, coefs), "layer_activation, delta, and coefs must align as one dense MLP layer")
@icontract.require(lambda alpha: _nonnegative_scalar(alpha), "alpha must be finite and nonnegative")
@icontract.require(lambda sample_weight_sum: _sample_weight_sum_valid(sample_weight_sum), "sample_weight_sum must be finite and nonnegative when provided")
@icontract.ensure(lambda result, coefs, delta: _layer_gradient_result_valid(result, coefs, delta), "layer gradients must match coefficient and intercept shapes")
def mlp_layer_gradients(
    layer_activation: NDArray[np.float64],
    delta: NDArray[np.float64],
    coefs: NDArray[np.float64],
    *,
    alpha: float = 0.0,
    sample_weight_sum: float | None = None,
) -> LayerGradient:
    """Compute one dense coefficient-gradient matrix and intercept-gradient vector."""
    activation_values = np.asarray(layer_activation, dtype=np.float64)
    delta_values = np.asarray(delta, dtype=np.float64)
    coef_values = np.asarray(coefs, dtype=np.float64)
    normalizer = float(sample_weight_sum) if sample_weight_sum is not None else float(activation_values.shape[0])
    coef_grad = activation_values.T.dot(delta_values)
    coef_grad += float(alpha) * coef_values
    coef_grad /= normalizer
    intercept_grad = np.sum(delta_values, axis=0) / normalizer
    return np.asarray(coef_grad, dtype=np.float64), np.asarray(intercept_grad, dtype=np.float64)


@register_atom(witness_mlp_backprop)
@icontract.require(lambda X, coefs, intercepts: _network_inputs_valid(X, coefs, intercepts), "X, coefs, and intercepts must define a finite dense MLP parameter chain")
@icontract.require(lambda y, X, coefs: _targets_valid(y, X, coefs), "y must be a finite dense target matrix aligned with the supplied network output")
@icontract.require(lambda hidden_activation: _hidden_activation_name_valid(hidden_activation), "hidden_activation must be one of sklearn's hidden-layer activations")
@icontract.require(lambda loss_name, output_activation: _canonical_loss_combo_valid(loss_name, output_activation), "loss_name and output_activation must form one sklearn MLP canonical output-loss pair")
@icontract.require(lambda y, loss_name, output_activation: loss_name != "log_loss" or _log_targets_valid(y, y, output_activation), "log-loss targets must already be bounded indicator values")
@icontract.require(lambda coefs, output_activation: output_activation != "softmax" or np.asarray(coefs[-1], dtype=np.float64).shape[1] >= 2, "softmax output layers must have at least two units")
@icontract.require(lambda alpha: _nonnegative_scalar(alpha), "alpha must be finite and nonnegative")
@icontract.require(lambda X, sample_weight: _sample_weight_valid(sample_weight, np.asarray(X, dtype=np.float64).shape[0]), "sample_weight must be a finite nonnegative vector over samples")
@icontract.ensure(lambda result, coefs, intercepts: _backprop_result_valid(result, coefs, intercepts), "backprop must return a finite loss with one gradient per supplied parameter tensor")
def mlp_backprop(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    coefs: WeightTuple,
    intercepts: BiasTuple,
    *,
    hidden_activation: HiddenActivationName,
    output_activation: ActivationName,
    loss_name: LossName,
    alpha: float = 0.0,
    sample_weight: NDArray[np.float64] | None = None,
) -> BackpropResult:
    """Compute dense MLP loss and parameter gradients for one fixed parameter state."""
    activations = mlp_forward_pass(
        X,
        coefs,
        intercepts,
        hidden_activation=hidden_activation,
        output_activation=output_activation,
    )
    prediction = activations[-1]
    loss = mlp_loss(y, prediction, loss_name=loss_name, output_activation=output_activation, sample_weight=sample_weight)

    n_samples = np.asarray(X, dtype=np.float64).shape[0]
    normalizer = float(np.sum(np.asarray(sample_weight, dtype=np.float64))) if sample_weight is not None else float(n_samples)
    weight_l2 = sum(float(np.dot(np.ravel(np.asarray(coef, dtype=np.float64)), np.ravel(np.asarray(coef, dtype=np.float64)))) for coef in coefs)
    loss += (0.5 * float(alpha)) * weight_l2 / normalizer

    last = len(coefs) - 1
    deltas: list[NDArray[np.float64]] = [np.zeros((n_samples, np.asarray(coef, dtype=np.float64).shape[1]), dtype=np.float64) for coef in coefs]
    coef_grads: list[NDArray[np.float64]] = [np.empty_like(np.asarray(coef, dtype=np.float64)) for coef in coefs]
    intercept_grads: list[NDArray[np.float64]] = [np.empty_like(np.asarray(intercept, dtype=np.float64)) for intercept in intercepts]

    deltas[last] = np.asarray(prediction - np.asarray(y, dtype=np.float64), dtype=np.float64)
    if sample_weight is not None:
        deltas[last] *= np.asarray(sample_weight, dtype=np.float64).reshape(-1, 1)
    coef_grads[last], intercept_grads[last] = mlp_layer_gradients(
        activations[last],
        deltas[last],
        coefs[last],
        alpha=alpha,
        sample_weight_sum=normalizer,
    )

    for index in range(last - 1, -1, -1):
        propagated = np.asarray(deltas[index + 1].dot(np.asarray(coefs[index + 1], dtype=np.float64).T), dtype=np.float64)
        deltas[index] = mlp_activation_derivative(activations[index + 1], propagated, activation=hidden_activation)
        coef_grads[index], intercept_grads[index] = mlp_layer_gradients(
            activations[index],
            deltas[index],
            coefs[index],
            alpha=alpha,
            sample_weight_sum=normalizer,
        )

    return float(loss), tuple(coef_grads), tuple(intercept_grads)
