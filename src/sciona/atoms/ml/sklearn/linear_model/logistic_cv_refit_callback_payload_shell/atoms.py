"""Sklearn LogisticRegressionCV refit callback payload atoms."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_logistic_cv_refit_first_weight,
    witness_logistic_cv_refit_path_call,
    witness_logistic_cv_refit_path_kwargs,
    witness_logistic_cv_refit_single_Cs,
    witness_logistic_cv_refit_verbose,
)

_MULTI_CLASS_VALUES = {"ovr", "multinomial"}


def _integral(value: object) -> bool:
    return bool(isinstance(value, Integral) and not isinstance(value, bool))


def _positive_integral(value: object) -> bool:
    return bool(_integral(value) and int(value) >= 1)


def _nonnegative_real(value: object) -> bool:
    return bool(isinstance(value, Real) and not isinstance(value, bool) and float(value) >= 0.0)


def _positive_real(value: object) -> bool:
    return bool(isinstance(value, Real) and not isinstance(value, bool) and float(value) > 0.0)


def _bool_value(value: object) -> bool:
    return bool(isinstance(value, bool))


def _multi_class_valid(value: object) -> bool:
    return bool(value in _MULTI_CLASS_VALUES)


def _kwargs_valid(kwargs: object) -> bool:
    expected_keys = {
        "pos_class",
        "Cs",
        "solver",
        "fit_intercept",
        "coef",
        "max_iter",
        "tol",
        "penalty",
        "class_weight",
        "multi_class",
        "verbose",
        "random_state",
        "check_input",
        "max_squared_sum",
        "sample_weight",
        "l1_ratio",
    }
    return bool(isinstance(kwargs, Mapping) and set(kwargs) == expected_keys)


def _sequence_has_first(value: object) -> bool:
    try:
        value[0]  # type: ignore[index]
        return bool(len(value) >= 1)  # type: ignore[arg-type]
    except (IndexError, KeyError, TypeError):
        return False


def _single_Cs_valid(result: list[object], C_value: object) -> bool:
    return bool(isinstance(result, list) and len(result) == 1 and result[0] is C_value)


def _verbose_valid(result: int, verbose: int) -> bool:
    return bool(isinstance(result, int) and result == max(0, int(verbose) - 1))


def _path_kwargs_valid(
    result: dict[str, object],
    pos_class: object,
    C_value: object,
    solver: object,
    fit_intercept: bool,
    coef_init: object,
    max_iter: int,
    tol: object,
    penalty: object,
    class_weight: object,
    multi_class: str,
    verbose: int,
    random_state: object,
    max_squared_sum: object,
    sample_weight: object,
    l1_ratio: object,
) -> bool:
    return bool(
        _kwargs_valid(result)
        and result["pos_class"] is pos_class
        and isinstance(result["Cs"], list)
        and len(result["Cs"]) == 1
        and result["Cs"][0] is C_value
        and result["solver"] is solver
        and result["fit_intercept"] is fit_intercept
        and result["coef"] is coef_init
        and result["max_iter"] is max_iter
        and result["tol"] is tol
        and result["penalty"] is penalty
        and result["class_weight"] is class_weight
        and result["multi_class"] == multi_class
        and result["verbose"] == max(0, int(verbose) - 1)
        and result["random_state"] is random_state
        and result["check_input"] is False
        and result["max_squared_sum"] is max_squared_sum
        and result["sample_weight"] is sample_weight
        and result["l1_ratio"] is l1_ratio
    )


def _path_call_valid(result: tuple[object, object, dict[str, object]], X: object, y: object, kwargs: object) -> bool:
    return bool(isinstance(result, tuple) and len(result) == 3 and result[0] is X and result[1] is y and result[2] is kwargs)


def _first_weight_valid(result: object, refit_weights: object) -> bool:
    expected = refit_weights[0]  # type: ignore[index]
    if isinstance(result, np.ndarray) or isinstance(expected, np.ndarray):
        return bool(np.array_equal(result, expected))
    return bool(result is expected or result == expected)


@register_atom(witness_logistic_cv_refit_single_Cs)
@icontract.require(lambda C_value: _positive_real(C_value), "C_value must be positive")
@icontract.ensure(lambda result, C_value: _single_Cs_valid(result, C_value), "refit Cs must be a one-element list containing C_value")
def logistic_cv_refit_single_Cs(C_value: object) -> list[object]:
    """Return the one-element C grid used for a LogisticRegressionCV refit."""
    return [C_value]


@register_atom(witness_logistic_cv_refit_verbose)
@icontract.require(lambda verbose: _integral(verbose), "verbose must be an integer")
@icontract.ensure(lambda result, verbose: _verbose_valid(result, verbose), "refit verbose must match max(0, verbose - 1)")
def logistic_cv_refit_verbose(verbose: int) -> int:
    """Return the decremented nonnegative verbose level used for the refit call."""
    return max(0, int(verbose) - 1)


@register_atom(witness_logistic_cv_refit_path_kwargs)
@icontract.require(lambda C_value: _positive_real(C_value), "C_value must be positive")
@icontract.require(lambda fit_intercept: _bool_value(fit_intercept), "fit_intercept must be boolean")
@icontract.require(lambda max_iter: _positive_integral(max_iter), "max_iter must be positive")
@icontract.require(lambda tol: _nonnegative_real(tol), "tol must be nonnegative")
@icontract.require(lambda multi_class: _multi_class_valid(multi_class), "multi_class must be ovr or multinomial")
@icontract.require(lambda verbose: _integral(verbose), "verbose must be an integer")
@icontract.ensure(
    lambda result, pos_class, C_value, solver, fit_intercept, coef_init, max_iter, tol, penalty, class_weight, multi_class, verbose, random_state, max_squared_sum, sample_weight, l1_ratio: _path_kwargs_valid(
        result,
        pos_class,
        C_value,
        solver,
        fit_intercept,
        coef_init,
        max_iter,
        tol,
        penalty,
        class_weight,
        multi_class,
        verbose,
        random_state,
        max_squared_sum,
        sample_weight,
        l1_ratio,
    ),
    "refit kwargs must match LogisticRegressionCV _logistic_regression_path call",
)
def logistic_cv_refit_path_kwargs(
    pos_class: object,
    C_value: object,
    solver: object,
    fit_intercept: bool,
    coef_init: object,
    max_iter: int,
    tol: object,
    penalty: object,
    class_weight: object,
    multi_class: str,
    verbose: int,
    random_state: object,
    max_squared_sum: object,
    sample_weight: object,
    l1_ratio: object,
) -> dict[str, object]:
    """Return keyword payload for the LogisticRegressionCV refit solver call."""
    return {
        "pos_class": pos_class,
        "Cs": [C_value],
        "solver": solver,
        "fit_intercept": fit_intercept,
        "coef": coef_init,
        "max_iter": max_iter,
        "tol": tol,
        "penalty": penalty,
        "class_weight": class_weight,
        "multi_class": multi_class,
        "verbose": max(0, int(verbose) - 1),
        "random_state": random_state,
        "check_input": False,
        "max_squared_sum": max_squared_sum,
        "sample_weight": sample_weight,
        "l1_ratio": l1_ratio,
    }


@register_atom(witness_logistic_cv_refit_path_call)
@icontract.require(lambda kwargs: _kwargs_valid(kwargs), "kwargs must match the refit solver keyword payload")
@icontract.ensure(lambda result, X, y, kwargs: _path_call_valid(result, X, y, kwargs), "refit call payload must preserve X, y, and kwargs identities")
def logistic_cv_refit_path_call(X: object, y: object, kwargs: dict[str, object]) -> tuple[object, object, dict[str, object]]:
    """Return positional and keyword payload for the LogisticRegressionCV refit solver call."""
    return (X, y, kwargs)


@register_atom(witness_logistic_cv_refit_first_weight)
@icontract.require(lambda refit_weights: _sequence_has_first(refit_weights), "refit_weights must contain at least one path result")
@icontract.ensure(lambda result, refit_weights: _first_weight_valid(result, refit_weights), "first weight must match sklearn w = w[0]")
def logistic_cv_refit_first_weight(refit_weights: object) -> object:
    """Return the first weight row from the refit solver-boundary result."""
    return refit_weights[0]  # type: ignore[index]
