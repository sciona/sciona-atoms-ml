"""Sklearn logistic scoring-path callback-shell atoms."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real

import icontract
import numpy as np
from numpy.typing import NDArray
from sklearn.utils.validation import _check_method_params

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_logistic_scoring_classes,
    witness_logistic_scoring_coef_intercept_state,
    witness_logistic_scoring_fold_split,
    witness_logistic_scoring_path_call,
    witness_logistic_scoring_path_kwargs,
    witness_logistic_scoring_positive_y_test,
    witness_logistic_scoring_result_tuple,
    witness_logistic_scoring_sample_weight_split,
    witness_logistic_scoring_score_call_payload,
    witness_logistic_scoring_score_params,
    witness_logistic_scoring_temp_log_reg_kwargs,
)

_MULTI_CLASS_VALUES = {"ovr", "multinomial"}
_PATH_KWARGS_KEYS = {
    "Cs",
    "l1_ratio",
    "fit_intercept",
    "solver",
    "max_iter",
    "class_weight",
    "pos_class",
    "multi_class",
    "tol",
    "verbose",
    "dual",
    "penalty",
    "intercept_scaling",
    "random_state",
    "check_input",
    "max_squared_sum",
    "sample_weight",
}


def _integral(value: object) -> bool:
    return bool(isinstance(value, Integral) and not isinstance(value, bool))


def _positive_integral(value: object) -> bool:
    return bool(_integral(value) and int(value) >= 1)


def _nonnegative_real(value: object) -> bool:
    return bool(isinstance(value, Real) and not isinstance(value, bool) and float(value) >= 0.0)


def _positive_real(value: object) -> bool:
    return bool(isinstance(value, Real) and not isinstance(value, bool) and float(value) > 0.0)


def _bool_value(value: object) -> bool:
    return bool(isinstance(value, (bool, np.bool_)))


def _multi_class_valid(value: object) -> bool:
    return bool(value in _MULTI_CLASS_VALUES)


def _indexing_valid(values: object, indices: object) -> bool:
    try:
        values[indices]  # type: ignore[index]
    except (IndexError, KeyError, TypeError, ValueError):
        return False
    return True


def _fold_split_valid(X: object, y: object, train: object, test: object) -> bool:
    return bool(
        _indexing_valid(X, train)
        and _indexing_valid(X, test)
        and _indexing_valid(y, train)
        and _indexing_valid(y, test)
    )


def _sample_weight_split_valid(sample_weight: object, train: object, test: object) -> bool:
    return bool(sample_weight is None or (_indexing_valid(sample_weight, train) and _indexing_valid(sample_weight, test)))


def _path_kwargs_valid(kwargs: object) -> bool:
    return bool(isinstance(kwargs, Mapping) and set(kwargs) == _PATH_KWARGS_KEYS and kwargs["check_input"] is False)


def _coef_state_input_valid(w: object, multi_class: str, fit_intercept: bool) -> bool:
    if not (_multi_class_valid(multi_class) and _bool_value(fit_intercept)):
        return False
    try:
        values = np.asarray(w)
    except (TypeError, ValueError):
        return False
    if values.size < 1 or not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
        return False
    if multi_class == "ovr":
        return bool(values.ndim == 1 and (not fit_intercept or values.shape[0] >= 2))
    return bool(values.ndim == 2 and values.shape[0] >= 1 and (not fit_intercept or values.shape[1] >= 2))


def _score_params_mapping(value: object) -> bool:
    return bool(value is None or isinstance(value, Mapping))


def _mapping_values_equal(left: Mapping[object, object], right: Mapping[object, object]) -> bool:
    if left.keys() != right.keys():
        return False
    for key in left:
        left_value = left[key]
        right_value = right[key]
        if isinstance(left_value, np.ndarray) or isinstance(right_value, np.ndarray):
            if not np.array_equal(left_value, right_value):
                return False
        elif left_value is not right_value and left_value != right_value:
            return False
    return True


def _fold_split_result_valid(result: tuple[object, object, object, object], X: object, y: object, train: object, test: object) -> bool:
    expected = (X[train], X[test], y[train], y[test])  # type: ignore[index]
    return bool(isinstance(result, tuple) and len(result) == 4 and all(np.array_equal(result[i], expected[i]) for i in range(4)))


def _sample_weight_result_valid(result: tuple[object, object], sample_weight: object, train: object, test: object) -> bool:
    if sample_weight is None:
        return bool(result == (None, None))
    expected = (sample_weight[train], sample_weight[test])  # type: ignore[index]
    return bool(isinstance(result, tuple) and len(result) == 2 and np.array_equal(result[0], expected[0]) and np.array_equal(result[1], expected[1]))


def _path_kwargs_result_valid(
    result: dict[str, object],
    Cs: object,
    l1_ratio: object,
    fit_intercept: bool,
    solver: object,
    max_iter: int,
    class_weight: object,
    pos_class: object,
    multi_class: str,
    tol: object,
    verbose: int,
    dual: bool,
    penalty: object,
    intercept_scaling: object,
    random_state: object,
    max_squared_sum: object,
    sw_train: object,
) -> bool:
    return bool(
        _path_kwargs_valid(result)
        and result["Cs"] is Cs
        and result["l1_ratio"] is l1_ratio
        and result["fit_intercept"] is fit_intercept
        and result["solver"] is solver
        and result["max_iter"] == int(max_iter)
        and result["class_weight"] is class_weight
        and result["pos_class"] is pos_class
        and result["multi_class"] == multi_class
        and result["tol"] is tol
        and result["verbose"] == int(verbose)
        and result["dual"] is dual
        and result["penalty"] is penalty
        and result["intercept_scaling"] is intercept_scaling
        and result["random_state"] is random_state
        and result["max_squared_sum"] is max_squared_sum
        and result["sample_weight"] is sw_train
    )


def _path_call_valid(result: tuple[object, object, dict[str, object]], X_train: object, y_train: object, kwargs: object) -> bool:
    return bool(isinstance(result, tuple) and len(result) == 3 and result[0] is X_train and result[1] is y_train and result[2] is kwargs)


def _classes_result_valid(result: NDArray[object], multi_class: str, y_train: object) -> bool:
    expected = np.array([-1, 1]) if multi_class == "ovr" else np.unique(y_train)
    return bool(isinstance(result, np.ndarray) and np.array_equal(result, expected))


def _positive_y_result_valid(result: object, y_test: object, pos_class: object) -> bool:
    if pos_class is None:
        return bool(result is y_test)
    y_values = np.asarray(y_test)
    expected = np.ones(y_values.shape, dtype=np.float64)
    expected[~(y_values == pos_class)] = -1.0
    return bool(isinstance(result, np.ndarray) and result.dtype == np.float64 and np.array_equal(result, expected))


def _coef_state_result_valid(result: tuple[object, object], w: object, multi_class: str, fit_intercept: bool) -> bool:
    values = np.asarray(w)
    path_row = values[np.newaxis, :] if multi_class == "ovr" else values
    expected_coef = path_row[:, :-1] if fit_intercept else path_row
    expected_intercept = path_row[:, -1] if fit_intercept else 0.0
    intercept_valid = np.array_equal(result[1], expected_intercept) if fit_intercept else result[1] == 0.0
    return bool(
        isinstance(result, tuple)
        and len(result) == 2
        and isinstance(result[0], np.ndarray)
        and np.array_equal(result[0], expected_coef)
        and intercept_valid
    )


def _score_params_result_valid(result: dict[object, object], X: object, score_params: object, test: object) -> bool:
    expected = _check_method_params(X=X, params={} if score_params is None else dict(score_params), indices=test)
    return bool(isinstance(result, dict) and _mapping_values_equal(result, expected))


def _score_payload_valid(result: tuple[object, ...], scoring: object, estimator_state: object, X_test: object, y_test: object, sw_test: object, score_params: object) -> bool:
    if scoring is None:
        return bool(
            isinstance(result, tuple)
            and len(result) == 5
            and result[0] == "estimator_score"
            and result[1] is estimator_state
            and result[2] is X_test
            and result[3] is y_test
            and isinstance(result[4], Mapping)
            and set(result[4]) == {"sample_weight"}
            and result[4]["sample_weight"] is sw_test
        )
    return bool(
        isinstance(result, tuple)
        and len(result) == 6
        and result[0] == "scorer"
        and result[1] is scoring
        and result[2] is estimator_state
        and result[3] is X_test
        and result[4] is y_test
        and isinstance(result[5], Mapping)
        and isinstance(score_params, Mapping)
        and _mapping_values_equal(result[5], score_params)
    )


def _result_tuple_valid(result: tuple[object, object, NDArray[object], object], coefs: object, Cs: object, scores: object, n_iter: object) -> bool:
    expected_scores = np.array(scores)
    return bool(
        isinstance(result, tuple)
        and len(result) == 4
        and result[0] is coefs
        and result[1] is Cs
        and isinstance(result[2], np.ndarray)
        and np.array_equal(result[2], expected_scores)
        and result[3] is n_iter
    )


@register_atom(witness_logistic_scoring_fold_split)
@icontract.require(lambda X, y, train, test: _fold_split_valid(X, y, train, test), "X and y must support train/test indexing")
@icontract.ensure(lambda result, X, y, train, test: _fold_split_result_valid(result, X, y, train, test), "fold split must match sklearn train/test slicing")
def logistic_scoring_fold_split(X: object, y: object, train: object, test: object) -> tuple[object, object, object, object]:
    """Return X_train, X_test, y_train, and y_test slices for a scoring fold."""
    return X[train], X[test], y[train], y[test]  # type: ignore[index]


@register_atom(witness_logistic_scoring_sample_weight_split)
@icontract.require(lambda sample_weight, train, test: _sample_weight_split_valid(sample_weight, train, test), "sample_weight must be None or support train/test indexing")
@icontract.ensure(lambda result, sample_weight, train, test: _sample_weight_result_valid(result, sample_weight, train, test), "sample-weight split must match sklearn train/test slicing")
def logistic_scoring_sample_weight_split(sample_weight: object, train: object, test: object) -> tuple[object, object]:
    """Return train and test sample-weight slices, preserving sklearn's None branch."""
    if sample_weight is None:
        return None, None
    return sample_weight[train], sample_weight[test]  # type: ignore[index]


@register_atom(witness_logistic_scoring_path_kwargs)
@icontract.require(lambda fit_intercept: _bool_value(fit_intercept), "fit_intercept must be boolean")
@icontract.require(lambda max_iter: _positive_integral(max_iter), "max_iter must be positive")
@icontract.require(lambda multi_class: _multi_class_valid(multi_class), "multi_class must be ovr or multinomial")
@icontract.require(lambda tol: _nonnegative_real(tol), "tol must be nonnegative")
@icontract.require(lambda verbose: _integral(verbose), "verbose must be an integer")
@icontract.require(lambda dual: _bool_value(dual), "dual must be boolean")
@icontract.require(lambda intercept_scaling: _positive_real(intercept_scaling), "intercept_scaling must be positive")
@icontract.ensure(
    lambda result, Cs, l1_ratio, fit_intercept, solver, max_iter, class_weight, pos_class, multi_class, tol, verbose, dual, penalty, intercept_scaling, random_state, max_squared_sum, sw_train: _path_kwargs_result_valid(
        result,
        Cs,
        l1_ratio,
        fit_intercept,
        solver,
        max_iter,
        class_weight,
        pos_class,
        multi_class,
        tol,
        verbose,
        dual,
        penalty,
        intercept_scaling,
        random_state,
        max_squared_sum,
        sw_train,
    ),
    "_logistic_regression_path kwargs must match sklearn scoring-path payload",
)
def logistic_scoring_path_kwargs(
    Cs: object,
    l1_ratio: object,
    fit_intercept: bool,
    solver: object,
    max_iter: int,
    class_weight: object,
    pos_class: object,
    multi_class: str,
    tol: object,
    verbose: int,
    dual: bool,
    penalty: object,
    intercept_scaling: object,
    random_state: object,
    max_squared_sum: object,
    sw_train: object,
) -> dict[str, object]:
    """Return keyword payload for the _logistic_regression_path solver boundary."""
    return {
        "Cs": Cs,
        "l1_ratio": l1_ratio,
        "fit_intercept": fit_intercept,
        "solver": solver,
        "max_iter": int(max_iter),
        "class_weight": class_weight,
        "pos_class": pos_class,
        "multi_class": multi_class,
        "tol": tol,
        "verbose": int(verbose),
        "dual": dual,
        "penalty": penalty,
        "intercept_scaling": intercept_scaling,
        "random_state": random_state,
        "check_input": False,
        "max_squared_sum": max_squared_sum,
        "sample_weight": sw_train,
    }


@register_atom(witness_logistic_scoring_path_call)
@icontract.require(lambda kwargs: _path_kwargs_valid(kwargs), "kwargs must match the scoring-path solver keyword payload")
@icontract.ensure(lambda result, X_train, y_train, kwargs: _path_call_valid(result, X_train, y_train, kwargs), "path call payload must preserve X_train, y_train, and kwargs identities")
def logistic_scoring_path_call(X_train: object, y_train: object, kwargs: dict[str, object]) -> tuple[object, object, dict[str, object]]:
    """Return positional and keyword payload for the deferred solver call."""
    return (X_train, y_train, kwargs)


@register_atom(witness_logistic_scoring_temp_log_reg_kwargs)
@icontract.require(lambda multi_class: _multi_class_valid(multi_class), "multi_class must be ovr or multinomial")
@icontract.ensure(lambda result, solver, multi_class: result == {"solver": solver, "multi_class": multi_class}, "temporary estimator kwargs must match LogisticRegression construction")
def logistic_scoring_temp_log_reg_kwargs(solver: object, multi_class: str) -> dict[str, object]:
    """Return constructor kwargs for the temporary LogisticRegression scorer object."""
    return {"solver": solver, "multi_class": multi_class}


@register_atom(witness_logistic_scoring_classes)
@icontract.require(lambda multi_class: _multi_class_valid(multi_class), "multi_class must be ovr or multinomial")
@icontract.ensure(lambda result, multi_class, y_train: _classes_result_valid(result, multi_class, y_train), "classes_ must match sklearn scoring-path assignment")
def logistic_scoring_classes(multi_class: str, y_train: object) -> NDArray[object]:
    """Return classes_ assigned to the temporary LogisticRegression scorer object."""
    if multi_class == "ovr":
        return np.array([-1, 1])
    return np.unique(y_train)


@register_atom(witness_logistic_scoring_positive_y_test)
@icontract.ensure(lambda result, y_test, pos_class: _positive_y_result_valid(result, y_test, pos_class), "positive-class recoding must match sklearn OvR scoring target")
def logistic_scoring_positive_y_test(y_test: object, pos_class: object) -> object:
    """Return y_test recoded to {-1.0, 1.0} when scoring a positive class."""
    if pos_class is None:
        return y_test
    y_values = np.asarray(y_test)
    recoded = np.ones(y_values.shape, dtype=np.float64)
    recoded[~(y_values == pos_class)] = -1.0
    return recoded


@register_atom(witness_logistic_scoring_coef_intercept_state)
@icontract.require(lambda w, multi_class, fit_intercept: _coef_state_input_valid(w, multi_class, fit_intercept), "coefficient row must match scoring-path layout")
@icontract.ensure(lambda result, w, multi_class, fit_intercept: _coef_state_result_valid(result, w, multi_class, fit_intercept), "coef_ and intercept_ state must match sklearn assignment")
def logistic_scoring_coef_intercept_state(w: object, multi_class: str, fit_intercept: bool) -> tuple[NDArray[np.floating], object]:
    """Return coef_ and intercept_ state assigned before a scoring callback."""
    path_row = np.asarray(w)
    if multi_class == "ovr":
        path_row = path_row[np.newaxis, :]
    if fit_intercept:
        return path_row[:, :-1], path_row[:, -1]
    return path_row, 0.0


@register_atom(witness_logistic_scoring_score_params)
@icontract.require(lambda score_params: _score_params_mapping(score_params), "score_params must be None or a mapping")
@icontract.ensure(lambda result, X, score_params, test: _score_params_result_valid(result, X, score_params, test), "score_params must match sklearn method-parameter validation")
def logistic_scoring_score_params(X: object, score_params: Mapping[object, object] | None, test: object) -> dict[object, object]:
    """Return score parameters sliced to the held-out fold when applicable."""
    return _check_method_params(X=X, params={} if score_params is None else dict(score_params), indices=test)


@register_atom(witness_logistic_scoring_score_call_payload)
@icontract.require(lambda score_params: isinstance(score_params, Mapping), "score_params must be a mapping")
@icontract.ensure(
    lambda result, scoring, estimator_state, X_test, y_test, sw_test, score_params: _score_payload_valid(
        result,
        scoring,
        estimator_state,
        X_test,
        y_test,
        sw_test,
        score_params,
    ),
    "score-call payload must match sklearn scorer branch without executing the scorer",
)
def logistic_scoring_score_call_payload(
    scoring: object,
    estimator_state: object,
    X_test: object,
    y_test: object,
    sw_test: object,
    score_params: Mapping[object, object],
) -> tuple[object, ...]:
    """Return the score or scorer call payload without executing it."""
    if scoring is None:
        return ("estimator_score", estimator_state, X_test, y_test, {"sample_weight": sw_test})
    return ("scorer", scoring, estimator_state, X_test, y_test, dict(score_params))


@register_atom(witness_logistic_scoring_result_tuple)
@icontract.ensure(lambda result, coefs, Cs, scores, n_iter: _result_tuple_valid(result, coefs, Cs, scores, n_iter), "result tuple must match _log_reg_scoring_path return packaging")
def logistic_scoring_result_tuple(coefs: object, Cs: object, scores: object, n_iter: object) -> tuple[object, object, NDArray[object], object]:
    """Return final _log_reg_scoring_path output tuple with scores converted to an ndarray."""
    return coefs, Cs, np.array(scores), n_iter
