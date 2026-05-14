"""Sklearn HuberRegressor fit optimizer-shell atoms."""

from __future__ import annotations

from collections.abc import Mapping

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_huber_fit_bounds,
    witness_huber_fit_initial_parameters,
    witness_huber_fit_optimizer_payload,
    witness_huber_fit_outlier_handoff_payload,
    witness_huber_fit_result_attributes,
    witness_huber_fit_status2_failure_message,
)

_OPTIMIZER_PAYLOAD_KEYS = {"fun", "x0", "method", "jac", "args", "options", "bounds"}
_OUTLIER_HANDOFF_KEYS = {"X", "y", "coef", "intercept", "scale", "epsilon"}


def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _bool_value(value: bool) -> bool:
    return isinstance(value, bool)


def _finite_scalar(value: float) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)))


def _positive_scalar(value: float) -> bool:
    return bool(_finite_scalar(value) and float(value) > 0.0)


def _epsilon_valid(value: float) -> bool:
    return bool(_finite_scalar(value) and float(value) >= 1.0)


def _alpha_valid(value: float) -> bool:
    return bool(_finite_scalar(value) and float(value) >= 0.0)


def _finite_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.size >= 1 and np.all(np.isfinite(array)))


def _coef_valid(coef: object, n_features: int) -> bool:
    if coef is None:
        return False
    try:
        values = np.asarray(coef, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 1 and values.shape[0] == n_features and np.all(np.isfinite(values)))


def _warm_start_inputs_valid(warm_start: bool, coef: object, n_features: int, intercept: float, scale: float) -> bool:
    if not warm_start:
        return True
    return bool(_coef_valid(coef, n_features) and _finite_scalar(intercept) and _positive_scalar(scale))


def _initial_parameters_valid(result: NDArray[np.float64], n_features: int, fit_intercept: bool, warm_start: bool) -> bool:
    values = np.asarray(result, dtype=np.float64)
    expected_size = n_features + 2 if warm_start or fit_intercept else n_features + 1
    return bool(values.ndim == 1 and values.shape[0] == expected_size and np.all(np.isfinite(values)) and values[-1] > 0.0)


def _bounds_valid(result: NDArray[np.float64], n_parameters: int) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(
        values.shape == (n_parameters, 2)
        and np.all(np.isneginf(values[:-1, 0]))
        and np.all(np.isposinf(values[:, 1]))
        and values[-1, 0] == np.finfo(np.float64).eps * 10
    )


def _parameters_valid(parameters: object) -> bool:
    return _finite_vector(parameters)


def _nonnegative_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 0)


def _fit_parameter_shape_valid(parameters: object, n_features: int, fit_intercept: bool) -> bool:
    values = np.asarray(parameters)
    if values.ndim != 1:
        return False
    if fit_intercept:
        return values.shape[0] == n_features + 2
    return values.shape[0] in {n_features + 1, n_features + 2}


def _fit_parameter_scale_valid(parameters: object) -> bool:
    values = np.asarray(parameters, dtype=np.float64)
    return bool(values.ndim == 1 and values.shape[0] >= 1 and values[-1] > 0.0)


def _optimizer_payload_valid(
    result: Mapping[str, object],
    objective: object,
    parameters: object,
    X: object,
    y: object,
    epsilon: float,
    alpha: float,
    sample_weight: object,
    max_iter: int,
    tol: float,
    bounds: object,
) -> bool:
    args = result.get("args")
    return bool(
        set(result) == _OPTIMIZER_PAYLOAD_KEYS
        and result["fun"] is objective
        and result["x0"] is parameters
        and result["method"] == "L-BFGS-B"
        and result["jac"] is True
        and isinstance(args, tuple)
        and len(args) == 5
        and args[0] is X
        and args[1] is y
        and args[2] == float(epsilon)
        and args[3] == float(alpha)
        and args[4] is sample_weight
        and result["options"] == {"maxiter": max_iter, "gtol": float(tol), "iprint": -1}
        and result["bounds"] is bounds
    )


def _result_attributes_valid(result: Mapping[str, object], parameters: object, n_features: int, fit_intercept: bool) -> bool:
    values = np.asarray(parameters, dtype=np.float64)
    coef = result.get("coef")
    return bool(
        _fit_parameter_shape_valid(parameters, n_features, fit_intercept)
        and set(result) == {"coef", "intercept", "scale"}
        and np.array_equal(np.asarray(coef, dtype=np.float64), values[:n_features])
        and result["scale"] == float(values[-1])
        and result["intercept"] == (float(values[-2]) if fit_intercept else 0.0)
    )


def _outlier_handoff_payload_valid(
    result: Mapping[str, object],
    X: object,
    y: object,
    coef: object,
    intercept: float,
    scale: float,
    epsilon: float,
) -> bool:
    return bool(
        set(result) == _OUTLIER_HANDOFF_KEYS
        and result["X"] is X
        and result["y"] is y
        and result["coef"] is coef
        and result["intercept"] == float(intercept)
        and result["scale"] == float(scale)
        and result["epsilon"] == float(epsilon)
    )


@register_atom(witness_huber_fit_initial_parameters)
@icontract.require(lambda n_features: _positive_int(n_features), "n_features must be positive")
@icontract.require(lambda fit_intercept: _bool_value(fit_intercept), "fit_intercept must be boolean")
@icontract.require(lambda warm_start: _bool_value(warm_start), "warm_start must be boolean")
@icontract.require(
    lambda warm_start, coef, n_features, intercept, scale: _warm_start_inputs_valid(
        warm_start,
        coef,
        n_features,
        intercept,
        scale,
    ),
    "warm-start parameters must include finite coef, intercept, and positive scale",
)
@icontract.ensure(
    lambda result, n_features, fit_intercept, warm_start: _initial_parameters_valid(
        result,
        n_features,
        fit_intercept,
        warm_start,
    ),
    "initial parameters must match sklearn HuberRegressor fit layout",
)
def huber_fit_initial_parameters(
    n_features: int,
    *,
    fit_intercept: bool,
    warm_start: bool,
    coef: NDArray[np.float64] | None = None,
    intercept: float = 0.0,
    scale: float = 1.0,
) -> NDArray[np.float64]:
    """Return HuberRegressor's initial optimizer parameter vector."""
    if warm_start:
        return np.concatenate((np.asarray(coef, dtype=np.float64), [float(intercept), float(scale)]))

    size = n_features + 2 if fit_intercept else n_features + 1
    parameters = np.zeros(size, dtype=np.float64)
    parameters[-1] = 1.0
    return parameters


@register_atom(witness_huber_fit_bounds)
@icontract.require(lambda n_parameters: _positive_int(n_parameters), "n_parameters must be positive")
@icontract.ensure(lambda result, n_parameters: _bounds_valid(result, n_parameters), "bounds must match sklearn Huber scale lower-bound layout")
def huber_fit_bounds(n_parameters: int) -> NDArray[np.float64]:
    """Return the L-BFGS-B bounds matrix used by HuberRegressor.fit."""
    bounds = np.tile([-np.inf, np.inf], (n_parameters, 1))
    bounds[-1][0] = np.finfo(np.float64).eps * 10
    return np.asarray(bounds, dtype=np.float64)


@register_atom(witness_huber_fit_optimizer_payload)
@icontract.require(lambda objective: objective is not None, "objective callback must be provided")
@icontract.require(lambda parameters: _parameters_valid(parameters), "parameters must be a finite vector")
@icontract.require(lambda X: X is not None, "X must be provided")
@icontract.require(lambda y: y is not None, "y must be provided")
@icontract.require(lambda epsilon: _epsilon_valid(epsilon), "epsilon must be at least one")
@icontract.require(lambda alpha: _alpha_valid(alpha), "alpha must be nonnegative")
@icontract.require(lambda max_iter: _nonnegative_int(max_iter), "max_iter must be nonnegative")
@icontract.require(lambda tol: _finite_scalar(tol), "tol must be finite")
@icontract.require(lambda bounds: bounds is not None, "bounds must be provided")
@icontract.ensure(
    lambda result, objective, parameters, X, y, epsilon, alpha, sample_weight, max_iter, tol, bounds: _optimizer_payload_valid(
        result,
        objective,
        parameters,
        X,
        y,
        epsilon,
        alpha,
        sample_weight,
        max_iter,
        tol,
        bounds,
    ),
    "optimizer payload must match HuberRegressor optimize.minimize call",
)
def huber_fit_optimizer_payload(
    objective: object,
    parameters: NDArray[np.float64],
    X: object,
    y: object,
    *,
    epsilon: float,
    alpha: float,
    sample_weight: object,
    max_iter: int,
    tol: float,
    bounds: object,
) -> dict[str, object]:
    """Return the optimize.minimize payload assembled by HuberRegressor.fit."""
    return {
        "fun": objective,
        "x0": parameters,
        "method": "L-BFGS-B",
        "jac": True,
        "args": (X, y, float(epsilon), float(alpha), sample_weight),
        "options": {"maxiter": max_iter, "gtol": float(tol), "iprint": -1},
        "bounds": bounds,
    }


@register_atom(witness_huber_fit_status2_failure_message)
@icontract.require(lambda status: isinstance(status, int) and not isinstance(status, bool), "status must be an integer")
@icontract.ensure(
    lambda result, status, message: (
        result == f"HuberRegressor convergence failed: l-BFGS-b solver terminated with {message}"
        if status == 2
        else result is None
    ),
    "status-2 message must match sklearn HuberRegressor wording",
)
def huber_fit_status2_failure_message(status: int, message: object) -> str | None:
    """Return the status-2 convergence failure message, or None otherwise."""
    if status != 2:
        return None
    return f"HuberRegressor convergence failed: l-BFGS-b solver terminated with {message}"


@register_atom(witness_huber_fit_result_attributes)
@icontract.require(lambda parameters: _parameters_valid(parameters), "parameters must be a finite vector")
@icontract.require(lambda n_features: _positive_int(n_features), "n_features must be positive")
@icontract.require(lambda fit_intercept: _bool_value(fit_intercept), "fit_intercept must be boolean")
@icontract.require(
    lambda parameters, n_features, fit_intercept: _fit_parameter_shape_valid(parameters, n_features, fit_intercept),
    "parameters must include coef, scale, and intercept when fitted with intercept",
)
@icontract.require(lambda parameters: _fit_parameter_scale_valid(parameters), "parameters must end with a positive scale")
@icontract.ensure(
    lambda result, parameters, n_features, fit_intercept: _result_attributes_valid(
        result,
        parameters,
        n_features,
        fit_intercept,
    ),
    "result attributes must match HuberRegressor optimizer-tail unpacking",
)
def huber_fit_result_attributes(
    parameters: NDArray[np.float64],
    n_features: int,
    *,
    fit_intercept: bool,
) -> dict[str, object]:
    """Return coef, intercept, and scale unpacked from optimizer parameters."""
    values = np.asarray(parameters, dtype=np.float64)
    return {
        "coef": values[:n_features],
        "intercept": float(values[-2]) if fit_intercept else 0.0,
        "scale": float(values[-1]),
    }


@register_atom(witness_huber_fit_outlier_handoff_payload)
@icontract.require(lambda X: X is not None, "X must be provided")
@icontract.require(lambda y: y is not None, "y must be provided")
@icontract.require(lambda coef: coef is not None, "coef must be provided")
@icontract.require(lambda intercept: _finite_scalar(intercept), "intercept must be finite")
@icontract.require(lambda scale: _positive_scalar(scale), "scale must be positive")
@icontract.require(lambda epsilon: _epsilon_valid(epsilon), "epsilon must be at least one")
@icontract.ensure(
    lambda result, X, y, coef, intercept, scale, epsilon: _outlier_handoff_payload_valid(
        result,
        X,
        y,
        coef,
        intercept,
        scale,
        epsilon,
    ),
    "outlier handoff payload must preserve fitted Huber atom inputs",
)
def huber_fit_outlier_handoff_payload(
    X: object,
    y: object,
    coef: object,
    *,
    intercept: float,
    scale: float,
    epsilon: float,
) -> dict[str, object]:
    """Return the postfit payload for existing Huber residual/outlier atoms."""
    return {
        "X": X,
        "y": y,
        "coef": coef,
        "intercept": float(intercept),
        "scale": float(scale),
        "epsilon": float(epsilon),
    }
