"""Dense MLP initialization helper atoms adapted from scikit-learn."""

from __future__ import annotations

from typing import Literal

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_mlp_glorot_init_bound,
    witness_mlp_init_layer_parameters,
    witness_mlp_initialize_parameters,
    witness_mlp_output_activation_name,
)

HiddenActivationName = Literal["identity", "logistic", "tanh", "relu"]
OutputActivationName = Literal["identity", "logistic", "softmax", "exp"]
LossName = Literal["squared_error", "poisson"]
LabelBinarizerType = Literal["binary", "multiclass", "multilabel-indicator"]
DtypeName = Literal["float32", "float64"]

LayerParameterInit = tuple[NDArray[np.floating], NDArray[np.floating]]
NetworkParameterInit = tuple[
    tuple[NDArray[np.floating], ...],
    tuple[NDArray[np.floating], ...],
    tuple[NDArray[np.floating], ...],
    tuple[NDArray[np.floating], ...],
]

_HIDDEN_ACTIVATIONS = {"identity", "logistic", "tanh", "relu"}
_OUTPUT_ACTIVATIONS = {"identity", "logistic", "softmax", "exp"}
_LOSS_NAMES = {"squared_error", "poisson"}
_LABEL_TYPES = {"binary", "multiclass", "multilabel-indicator"}
_DTYPE_NAMES = {"float32", "float64"}


def _hidden_activation_valid(activation: str) -> bool:
    return bool(isinstance(activation, str) and activation in _HIDDEN_ACTIVATIONS)


def _loss_name_valid(loss_name: str) -> bool:
    return bool(isinstance(loss_name, str) and loss_name in _LOSS_NAMES)


def _label_type_valid(label_binarizer_type: str | None) -> bool:
    return bool(label_binarizer_type is None or (isinstance(label_binarizer_type, str) and label_binarizer_type in _LABEL_TYPES))


def _dtype_name_valid(dtype_name: str) -> bool:
    return bool(isinstance(dtype_name, str) and dtype_name in _DTYPE_NAMES)


def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _layer_units_valid(layer_units: tuple[int, ...]) -> bool:
    return bool(isinstance(layer_units, tuple) and len(layer_units) >= 2 and all(_positive_int(unit) for unit in layer_units))


def _classifier_combo_valid(is_classifier: bool, loss_name: str, label_binarizer_type: str | None) -> bool:
    if bool(is_classifier):
        return bool(loss_name == "squared_error" and label_binarizer_type in _LABEL_TYPES)
    return bool(_loss_name_valid(loss_name) and label_binarizer_type is None)


def _glorot_bound_result_valid(result: float, fan_in: int, fan_out: int, activation: str) -> bool:
    factor = 2.0 if activation == "logistic" else 6.0
    expected = np.sqrt(factor / float(fan_in + fan_out))
    return bool(np.isfinite(float(result)) and float(result) > 0.0 and np.isclose(float(result), expected))


def _dtype(dtype_name: str) -> np.dtype[np.floating]:
    return np.dtype(dtype_name)


def _layer_init_result_valid(
    result: LayerParameterInit,
    fan_in: int,
    fan_out: int,
    dtype_name: str,
) -> bool:
    coefs, intercepts = result
    coef_values = np.asarray(coefs)
    intercept_values = np.asarray(intercepts)
    expected_dtype = _dtype(dtype_name)
    return bool(
        coef_values.shape == (fan_in, fan_out)
        and intercept_values.shape == (fan_out,)
        and coef_values.dtype == expected_dtype
        and intercept_values.dtype == expected_dtype
        and np.all(np.isfinite(coef_values))
        and np.all(np.isfinite(intercept_values))
    )


def _network_init_result_valid(
    result: NetworkParameterInit,
    layer_units: tuple[int, ...],
    dtype_name: str,
) -> bool:
    coefs, intercepts, best_coefs, best_intercepts = result
    expected_count = len(layer_units) - 1
    expected_dtype = _dtype(dtype_name)
    if not (
        isinstance(coefs, tuple)
        and isinstance(intercepts, tuple)
        and isinstance(best_coefs, tuple)
        and isinstance(best_intercepts, tuple)
        and len(coefs) == expected_count
        and len(intercepts) == expected_count
        and len(best_coefs) == expected_count
        and len(best_intercepts) == expected_count
    ):
        return False
    for index in range(expected_count):
        coef = np.asarray(coefs[index])
        intercept = np.asarray(intercepts[index])
        best_coef = np.asarray(best_coefs[index])
        best_intercept = np.asarray(best_intercepts[index])
        if coef.shape != (layer_units[index], layer_units[index + 1]) or intercept.shape != (layer_units[index + 1],):
            return False
        if coef.dtype != expected_dtype or intercept.dtype != expected_dtype:
            return False
        if best_coef.shape != coef.shape or best_intercept.shape != intercept.shape:
            return False
        if best_coef.dtype != expected_dtype or best_intercept.dtype != expected_dtype:
            return False
        if not (np.all(np.isfinite(coef)) and np.all(np.isfinite(intercept))):
            return False
        if not (np.array_equal(coef, best_coef) and np.array_equal(intercept, best_intercept)):
            return False
    return True


@register_atom(witness_mlp_output_activation_name)
@icontract.require(lambda loss_name: _loss_name_valid(loss_name), "loss_name must be squared_error or poisson")
@icontract.require(lambda label_binarizer_type: _label_type_valid(label_binarizer_type), "label_binarizer_type must be one of sklearn's label-binarizer target types when provided")
@icontract.require(lambda is_classifier, loss_name, label_binarizer_type: _classifier_combo_valid(is_classifier, loss_name, label_binarizer_type), "classifiers require a label-binarizer type and regressors require label_binarizer_type=None")
@icontract.ensure(lambda result: isinstance(result, str) and result in _OUTPUT_ACTIVATIONS, "output activation must be one of sklearn's MLP output activations")
def mlp_output_activation_name(
    *,
    is_classifier: bool,
    loss_name: LossName = "squared_error",
    label_binarizer_type: LabelBinarizerType | None = None,
) -> OutputActivationName:
    """Choose the MLP output activation name that sklearn derives during initialization."""
    if not is_classifier:
        return "exp" if loss_name == "poisson" else "identity"
    if label_binarizer_type == "multiclass":
        return "softmax"
    return "logistic"


@register_atom(witness_mlp_glorot_init_bound)
@icontract.require(lambda fan_in: _positive_int(fan_in), "fan_in must be a positive integer")
@icontract.require(lambda fan_out: _positive_int(fan_out), "fan_out must be a positive integer")
@icontract.require(lambda activation: _hidden_activation_valid(activation), "activation must be one of sklearn's hidden-layer MLP activations")
@icontract.ensure(lambda result, fan_in, fan_out, activation: _glorot_bound_result_valid(result, fan_in, fan_out, activation), "initialization bound must match sklearn's Glorot-style formula")
def mlp_glorot_init_bound(
    fan_in: int,
    fan_out: int,
    *,
    activation: HiddenActivationName,
) -> float:
    """Compute the layer initialization bound used by sklearn's MLP parameter initializer."""
    factor = 2.0 if activation == "logistic" else 6.0
    return float(np.sqrt(factor / float(fan_in + fan_out)))


@register_atom(witness_mlp_init_layer_parameters)
@icontract.require(lambda fan_in: _positive_int(fan_in), "fan_in must be a positive integer")
@icontract.require(lambda fan_out: _positive_int(fan_out), "fan_out must be a positive integer")
@icontract.require(lambda activation: _hidden_activation_valid(activation), "activation must be one of sklearn's hidden-layer MLP activations")
@icontract.require(lambda dtype_name: _dtype_name_valid(dtype_name), "dtype_name must be float32 or float64")
@icontract.ensure(lambda result, fan_in, fan_out, dtype_name: _layer_init_result_valid(result, fan_in, fan_out, dtype_name), "layer parameter tensors must match the requested shapes and dtype")
def mlp_init_layer_parameters(
    fan_in: int,
    fan_out: int,
    *,
    activation: HiddenActivationName,
    random_state: int | None = None,
    dtype_name: DtypeName = "float64",
) -> LayerParameterInit:
    """Draw one coefficient matrix and intercept vector using sklearn's MLP initialization rule."""
    rng = np.random.RandomState(random_state)
    init_bound = mlp_glorot_init_bound(fan_in, fan_out, activation=activation)
    dtype = _dtype(dtype_name)
    coef_init = rng.uniform(-init_bound, init_bound, (fan_in, fan_out)).astype(dtype, copy=False)
    intercept_init = rng.uniform(-init_bound, init_bound, fan_out).astype(dtype, copy=False)
    return coef_init, intercept_init


@register_atom(witness_mlp_initialize_parameters)
@icontract.require(lambda layer_units: _layer_units_valid(layer_units), "layer_units must be a tuple of at least two positive layer widths")
@icontract.require(lambda activation: _hidden_activation_valid(activation), "activation must be one of sklearn's hidden-layer MLP activations")
@icontract.require(lambda dtype_name: _dtype_name_valid(dtype_name), "dtype_name must be float32 or float64")
@icontract.ensure(lambda result, layer_units, dtype_name: _network_init_result_valid(result, layer_units, dtype_name), "network parameter tuples must align with layer_units and preserve best-copy mirrors")
def mlp_initialize_parameters(
    layer_units: tuple[int, ...],
    *,
    activation: HiddenActivationName,
    random_state: int | None = None,
    dtype_name: DtypeName = "float64",
) -> NetworkParameterInit:
    """Initialize all coefficient and intercept tensors for an MLP with supplied layer widths."""
    rng = np.random.RandomState(random_state)
    dtype = _dtype(dtype_name)
    coefs: list[NDArray[np.floating]] = []
    intercepts: list[NDArray[np.floating]] = []
    for index in range(len(layer_units) - 1):
        fan_in = layer_units[index]
        fan_out = layer_units[index + 1]
        init_bound = mlp_glorot_init_bound(fan_in, fan_out, activation=activation)
        coefs.append(rng.uniform(-init_bound, init_bound, (fan_in, fan_out)).astype(dtype, copy=False))
        intercepts.append(rng.uniform(-init_bound, init_bound, fan_out).astype(dtype, copy=False))
    coef_tuple = tuple(coefs)
    intercept_tuple = tuple(intercepts)
    best_coefs = tuple(np.array(coef, copy=True) for coef in coef_tuple)
    best_intercepts = tuple(np.array(intercept, copy=True) for intercept in intercept_tuple)
    return coef_tuple, intercept_tuple, best_coefs, best_intercepts
