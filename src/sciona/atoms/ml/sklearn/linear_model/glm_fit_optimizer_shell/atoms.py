"""Sklearn GLM fit optimizer-shell atoms."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_glm_fit_initial_coef,
    witness_glm_fit_intercept_init_value,
    witness_glm_fit_lbfgs_optimizer_payload,
    witness_glm_fit_newton_solver_payload,
    witness_glm_fit_result_attributes,
)

_LBFGS_PAYLOAD_KEYS = {"fun", "x0", "method", "jac", "options", "args"}
_NEWTON_PAYLOAD_KEYS = {
    "solver_class",
    "coef",
    "linear_loss",
    "l2_reg_strength",
    "tol",
    "max_iter",
    "n_threads",
}
_RESULT_ATTRIBUTE_KEYS = {"coef", "intercept"}


def _positive_integer(value: object) -> bool:
    return bool(isinstance(value, Integral) and not isinstance(value, bool) and int(value) > 0)


def _nonnegative_integer(value: object) -> bool:
    return bool(isinstance(value, Integral) and not isinstance(value, bool) and int(value) >= 0)


def _bool_value(value: object) -> bool:
    return bool(isinstance(value, (bool, np.bool_)))


def _finite_scalar(value: object) -> bool:
    return bool(isinstance(value, Real) and not isinstance(value, bool) and np.isfinite(float(value)))


def _positive_scalar(value: object) -> bool:
    return bool(_finite_scalar(value) and float(value) > 0.0)


def _nonnegative_scalar(value: object) -> bool:
    return bool(_finite_scalar(value) and float(value) >= 0.0)


def _float_dtype(value: object) -> bool:
    try:
        dtype = np.dtype(value)
    except TypeError:
        return False
    return bool(dtype in (np.dtype(np.float32), np.dtype(np.float64)))


def _finite_vector(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.size >= 1 and np.all(np.isfinite(array)))


def _warm_start_inputs_valid(
    warm_start: bool,
    coef: object,
    n_features: int,
    intercept: float,
) -> bool:
    if not warm_start:
        return True
    try:
        values = np.asarray(coef, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 1 and values.shape[0] == int(n_features) and np.all(np.isfinite(values)) and _finite_scalar(intercept))


def _sample_weight_valid(sample_weight: object, n_samples: int) -> bool:
    if sample_weight is None:
        return True
    try:
        weights = np.asarray(sample_weight, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(weights.ndim == 1 and weights.shape[0] == int(n_samples) and np.all(np.isfinite(weights)) and np.all(weights >= 0.0) and np.sum(weights) > 0.0)


def _target_valid(y: object) -> bool:
    try:
        values = np.asarray(y, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 1 and values.size >= 1 and np.all(np.isfinite(values)))


def _base_loss_has_link(base_loss: object) -> bool:
    return bool(callable(getattr(getattr(base_loss, "link", None), "link", None)))


def _intercept_link_value_finite(base_loss: object, y: object, sample_weight: object) -> bool:
    try:
        average = np.average(np.asarray(y, dtype=np.float64), weights=None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64))
        value = base_loss.link.link(average)
    except (AttributeError, TypeError, ValueError, ZeroDivisionError, FloatingPointError):
        return False
    return bool(np.isfinite(float(value)))


def _initial_coef_result_valid(
    result: NDArray[np.floating],
    n_features: int,
    fit_intercept: bool,
    warm_start: bool,
    loss_dtype: object,
    coef: object,
    intercept: float,
    intercept_init: float,
) -> bool:
    values = np.asarray(result)
    expected_size = int(n_features) + int(bool(fit_intercept))
    if values.ndim != 1 or values.shape[0] != expected_size or values.dtype != np.dtype(loss_dtype):
        return False
    if not np.all(np.isfinite(values)):
        return False
    if warm_start:
        expected = np.asarray(coef, dtype=loss_dtype)
        if fit_intercept:
            expected = np.concatenate((expected, np.asarray([intercept], dtype=loss_dtype)))
        return bool(np.array_equal(values, expected))
    if fit_intercept:
        return bool(np.all(values[:-1] == 0) and values[-1] == np.asarray(intercept_init, dtype=loss_dtype))
    return bool(np.all(values == 0))


def _lbfgs_payload_valid(
    result: Mapping[str, object],
    objective: object,
    coef: object,
    X: object,
    y: object,
    sample_weight: object,
    l2_reg_strength: float,
    n_threads: int,
    max_iter: int,
    tol: float,
    verbose: int,
) -> bool:
    args = result.get("args")
    options = result.get("options")
    return bool(
        set(result) == _LBFGS_PAYLOAD_KEYS
        and result["fun"] is objective
        and result["x0"] is coef
        and result["method"] == "L-BFGS-B"
        and result["jac"] is True
        and isinstance(args, tuple)
        and len(args) == 5
        and args[0] is X
        and args[1] is y
        and args[2] is sample_weight
        and args[3] == float(l2_reg_strength)
        and args[4] == int(n_threads)
        and options == {
            "maxiter": int(max_iter),
            "maxls": 50,
            "iprint": int(verbose) - 1,
            "gtol": float(tol),
            "ftol": 64 * np.finfo(float).eps,
        }
    )


def _newton_payload_valid(
    result: Mapping[str, object],
    solver_class: object,
    coef: object,
    linear_loss: object,
    l2_reg_strength: float,
    tol: float,
    max_iter: int,
    n_threads: int,
    verbose: int | None,
) -> bool:
    expected_keys = set(_NEWTON_PAYLOAD_KEYS)
    if verbose is not None:
        expected_keys.add("verbose")
    return bool(
        set(result) == expected_keys
        and result["solver_class"] is solver_class
        and result["coef"] is coef
        and result["linear_loss"] is linear_loss
        and result["l2_reg_strength"] == float(l2_reg_strength)
        and result["tol"] == float(tol)
        and result["max_iter"] == int(max_iter)
        and result["n_threads"] == int(n_threads)
        and (verbose is None or result["verbose"] == int(verbose))
    )


def _result_attributes_valid(result: Mapping[str, object], coef: object, fit_intercept: bool) -> bool:
    values = np.asarray(coef, dtype=np.float64)
    result_coef = np.asarray(result.get("coef"), dtype=np.float64)
    return bool(
        set(result) == _RESULT_ATTRIBUTE_KEYS
        and values.ndim == 1
        and values.size >= (2 if fit_intercept else 1)
        and np.all(np.isfinite(values))
        and result["intercept"] == (float(values[-1]) if fit_intercept else 0.0)
        and np.array_equal(result_coef, values[:-1] if fit_intercept else values)
    )


@register_atom(witness_glm_fit_initial_coef)
@icontract.require(lambda n_features: _positive_integer(n_features), "n_features must be positive")
@icontract.require(lambda fit_intercept: _bool_value(fit_intercept), "fit_intercept must be boolean")
@icontract.require(lambda warm_start: _bool_value(warm_start), "warm_start must be boolean")
@icontract.require(lambda loss_dtype: _float_dtype(loss_dtype), "loss_dtype must be float32 or float64")
@icontract.require(
    lambda warm_start, coef, n_features, intercept: _warm_start_inputs_valid(
        bool(warm_start),
        coef,
        n_features,
        intercept,
    ),
    "warm-start coefficients must be finite and align with n_features",
)
@icontract.require(lambda intercept_init: _finite_scalar(intercept_init), "intercept_init must be finite")
@icontract.ensure(
    lambda result, n_features, fit_intercept, warm_start, loss_dtype, coef, intercept, intercept_init: _initial_coef_result_valid(
        result,
        n_features,
        fit_intercept,
        warm_start,
        loss_dtype,
        coef,
        intercept,
        intercept_init,
    ),
    "initial coef must match sklearn GLM fit layout",
)
def glm_fit_initial_coef(
    n_features: int,
    *,
    fit_intercept: bool,
    warm_start: bool,
    loss_dtype: object,
    coef: object | None = None,
    intercept: float = 0.0,
    intercept_init: float = 0.0,
) -> NDArray[np.floating]:
    """Return the initial GLM fit coefficient vector before optimizer dispatch."""
    dtype = np.dtype(loss_dtype)
    if warm_start:
        values = np.asarray(coef, dtype=dtype)
        if fit_intercept:
            values = np.concatenate((values, np.asarray([intercept], dtype=dtype)))
        return values.astype(dtype, copy=False)

    values = np.zeros(int(n_features) + int(bool(fit_intercept)), dtype=dtype)
    if fit_intercept:
        values[-1] = np.asarray(intercept_init, dtype=dtype)
    return values


@register_atom(witness_glm_fit_intercept_init_value)
@icontract.require(lambda base_loss: _base_loss_has_link(base_loss), "base_loss must expose a link.link callable")
@icontract.require(lambda y: _target_valid(y), "y must be a finite 1D vector")
@icontract.require(lambda sample_weight, y: _sample_weight_valid(sample_weight, np.asarray(y).shape[0]), "sample_weight must be nonnegative and align with y")
@icontract.require(lambda base_loss, y, sample_weight: _intercept_link_value_finite(base_loss, y, sample_weight), "link of weighted target average must be finite")
@icontract.ensure(lambda result: _finite_scalar(result), "intercept initialization must be finite")
def glm_fit_intercept_init_value(base_loss: object, y: object, sample_weight: object | None = None) -> float:
    """Return the cold-start intercept value from the loss link of average y."""
    average = np.average(np.asarray(y, dtype=np.float64), weights=None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64))
    return float(base_loss.link.link(average))


@register_atom(witness_glm_fit_lbfgs_optimizer_payload)
@icontract.require(lambda objective: objective is not None, "objective callback must be provided")
@icontract.require(lambda coef: _finite_vector(coef), "coef must be a finite vector")
@icontract.require(lambda X: X is not None, "X must be provided")
@icontract.require(lambda y: y is not None, "y must be provided")
@icontract.require(lambda l2_reg_strength: _nonnegative_scalar(l2_reg_strength), "l2_reg_strength must be nonnegative")
@icontract.require(lambda n_threads: _positive_integer(n_threads), "n_threads must be positive")
@icontract.require(lambda max_iter: _nonnegative_integer(max_iter), "max_iter must be nonnegative")
@icontract.require(lambda tol: _positive_scalar(tol), "tol must be positive")
@icontract.require(lambda verbose: _nonnegative_integer(verbose), "verbose must be a nonnegative integer")
@icontract.ensure(
    lambda result, objective, coef, X, y, sample_weight, l2_reg_strength, n_threads, max_iter, tol, verbose: _lbfgs_payload_valid(
        result,
        objective,
        coef,
        X,
        y,
        sample_weight,
        l2_reg_strength,
        n_threads,
        max_iter,
        tol,
        verbose,
    ),
    "L-BFGS-B payload must match sklearn GLM fit optimizer call",
)
def glm_fit_lbfgs_optimizer_payload(
    objective: object,
    coef: object,
    X: object,
    y: object,
    *,
    sample_weight: object,
    l2_reg_strength: float,
    n_threads: int,
    max_iter: int,
    tol: float,
    verbose: int,
) -> dict[str, object]:
    """Return the scipy.optimize.minimize payload assembled by GLM fit."""
    return {
        "fun": objective,
        "x0": coef,
        "method": "L-BFGS-B",
        "jac": True,
        "options": {
            "maxiter": int(max_iter),
            "maxls": 50,
            "iprint": int(verbose) - 1,
            "gtol": float(tol),
            "ftol": 64 * np.finfo(float).eps,
        },
        "args": (X, y, sample_weight, float(l2_reg_strength), int(n_threads)),
    }


@register_atom(witness_glm_fit_newton_solver_payload)
@icontract.require(lambda solver_class: solver_class is not None, "solver_class must be provided")
@icontract.require(lambda coef: _finite_vector(coef), "coef must be a finite vector")
@icontract.require(lambda linear_loss: linear_loss is not None, "linear_loss must be provided")
@icontract.require(lambda l2_reg_strength: _nonnegative_scalar(l2_reg_strength), "l2_reg_strength must be nonnegative")
@icontract.require(lambda tol: _positive_scalar(tol), "tol must be positive")
@icontract.require(lambda max_iter: _nonnegative_integer(max_iter), "max_iter must be nonnegative")
@icontract.require(lambda n_threads: _positive_integer(n_threads), "n_threads must be positive")
@icontract.require(lambda verbose: verbose is None or _nonnegative_integer(verbose), "verbose must be None or a nonnegative integer")
@icontract.ensure(
    lambda result, solver_class, coef, linear_loss, l2_reg_strength, tol, max_iter, n_threads, verbose: _newton_payload_valid(
        result,
        solver_class,
        coef,
        linear_loss,
        l2_reg_strength,
        tol,
        max_iter,
        n_threads,
        verbose,
    ),
    "Newton solver payload must match sklearn GLM fit solver constructor",
)
def glm_fit_newton_solver_payload(
    solver_class: object,
    coef: object,
    linear_loss: object,
    *,
    l2_reg_strength: float,
    tol: float,
    max_iter: int,
    n_threads: int,
    verbose: int | None = None,
) -> dict[str, object]:
    """Return the Newton solver constructor payload assembled by GLM fit."""
    payload = {
        "solver_class": solver_class,
        "coef": coef,
        "linear_loss": linear_loss,
        "l2_reg_strength": float(l2_reg_strength),
        "tol": float(tol),
        "max_iter": int(max_iter),
        "n_threads": int(n_threads),
    }
    if verbose is not None:
        payload["verbose"] = int(verbose)
    return payload


@register_atom(witness_glm_fit_result_attributes)
@icontract.require(lambda coef: _finite_vector(coef), "coef must be a finite vector")
@icontract.require(lambda fit_intercept: _bool_value(fit_intercept), "fit_intercept must be boolean")
@icontract.require(lambda coef, fit_intercept: np.asarray(coef).ndim == 1 and np.asarray(coef).shape[0] >= (2 if fit_intercept else 1), "coef must have enough entries for requested intercept layout")
@icontract.ensure(lambda result, coef, fit_intercept: _result_attributes_valid(result, coef, fit_intercept), "result attributes must match GLM fit tail unpacking")
def glm_fit_result_attributes(coef: object, *, fit_intercept: bool) -> dict[str, object]:
    """Return final GLM coef/intercept attributes unpacked from an optimizer vector."""
    values = np.asarray(coef, dtype=np.float64)
    if fit_intercept:
        return {"intercept": float(values[-1]), "coef": values[:-1]}
    return {"intercept": 0.0, "coef": values}
