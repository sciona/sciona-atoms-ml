"""Public logistic sklearn API-shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_logistic_public_estimator_catalog,
    witness_logistic_public_estimator_family,
    witness_logistic_public_estimator_methods,
    witness_logistic_public_fit_method_payload,
    witness_logistic_public_fit_return_self,
    witness_logistic_public_fitted_state_summary,
    witness_logistic_public_prediction_method_payload,
    witness_logistic_public_solver_boundary,
)

_ESTIMATOR_NAMES = ("LogisticRegression", "LogisticRegressionCV")
_METHODS = ("fit", "predict", "predict_proba", "predict_log_proba", "decision_function", "score")
_SOLVERS = {"lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"}
_BOUNDARIES = {"liblinear_native", "scipy_or_sklearn_newton_optimizer", "sag_saga_native"}


def _known_estimator_name(value: object) -> bool:
    return value in _ESTIMATOR_NAMES


def _known_solver(value: object) -> bool:
    return value in _SOLVERS


def _finite_dense_matrix(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _target_vector_valid(y: object, X: object) -> bool:
    try:
        targets = np.asarray(y)
        matrix = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(matrix.ndim == 2 and targets.ndim == 1 and targets.shape == (matrix.shape[0],) and np.unique(targets).shape[0] >= 2)


def _sample_weight_valid(sample_weight: object, X: object) -> bool:
    if sample_weight is None:
        return True
    try:
        weights = np.asarray(sample_weight, dtype=np.float64)
        matrix = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(weights.ndim == 1 and matrix.ndim == 2 and weights.shape == (matrix.shape[0],) and np.all(np.isfinite(weights)) and np.all(weights >= 0.0))


def _fit_params_valid(estimator: object, params: object) -> bool:
    if params is None:
        return True
    if estimator.__class__.__name__ != "LogisticRegressionCV" or not isinstance(params, dict):
        return False
    return all(isinstance(key, str) and key != "" for key in params)


def _public_logistic_estimator(estimator: object) -> bool:
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

    return isinstance(estimator, (LogisticRegression, LogisticRegressionCV))


def _fitted_public_logistic(estimator: object) -> bool:
    return bool(
        _public_logistic_estimator(estimator)
        and hasattr(estimator, "classes_")
        and hasattr(estimator, "coef_")
        and hasattr(estimator, "intercept_")
        and hasattr(estimator, "n_features_in_")
        and hasattr(estimator, "n_iter_")
    )


def _fit_input_valid(estimator: object, X: object, y: object) -> bool:
    return bool(_public_logistic_estimator(estimator) and _finite_dense_matrix(X) and _target_vector_valid(y, X))


def _method_available(estimator: object, method_name: str) -> bool:
    return bool(isinstance(method_name, str) and method_name != "" and hasattr(estimator, method_name))


def _fit_payload_valid(result: object, estimator: object, X: object, y: object) -> bool:
    if not isinstance(result, dict):
        return False
    args = result.get("args")
    kwargs = result.get("kwargs")
    return bool(
        result.get("estimator") is estimator
        and result.get("method_name") == "fit"
        and isinstance(args, tuple)
        and len(args) == 2
        and args[0] is X
        and args[1] is y
        and isinstance(kwargs, dict)
    )


def _prediction_payload_valid(result: object, estimator: object, method_name: str, X: object) -> bool:
    if not isinstance(result, dict):
        return False
    args = result.get("args")
    return bool(
        result.get("estimator") is estimator
        and result.get("method_name") == method_name
        and isinstance(args, tuple)
        and len(args) == 1
        and args[0] is X
        and result.get("kwargs") == {}
    )


def _finite_array(value: object) -> bool:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return bool(array.size >= 1 and np.all(np.isfinite(array)))


@register_atom(witness_logistic_public_estimator_catalog)
@icontract.require(lambda catalog_scope: catalog_scope == "public_estimators", "catalog_scope must be 'public_estimators'")
@icontract.ensure(lambda result: result == _ESTIMATOR_NAMES, "catalog must expose covered public logistic estimators")
def logistic_public_estimator_catalog(catalog_scope: str = "public_estimators") -> tuple[str, ...]:
    """Expose public logistic estimator names for framework selection."""
    del catalog_scope
    return _ESTIMATOR_NAMES


@register_atom(witness_logistic_public_estimator_family)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered public logistic estimator")
@icontract.ensure(lambda result: result in {"logistic", "logistic_cv"}, "family must be logistic or logistic_cv")
def logistic_public_estimator_family(estimator_name: str) -> str:
    """Return the public logistic estimator family."""
    return "logistic_cv" if estimator_name == "LogisticRegressionCV" else "logistic"


@register_atom(witness_logistic_public_solver_boundary)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered public logistic estimator")
@icontract.require(lambda solver: _known_solver(solver), "solver must be an exposed logistic solver")
@icontract.ensure(lambda result: result in _BOUNDARIES, "boundary must name a covered logistic solver family")
def logistic_public_solver_boundary(estimator_name: str, solver: str) -> str:
    """Return the optimizer or native solver boundary selected for logistic fit."""
    del estimator_name
    if solver == "liblinear":
        return "liblinear_native"
    if solver in {"sag", "saga"}:
        return "sag_saga_native"
    return "scipy_or_sklearn_newton_optimizer"


@register_atom(witness_logistic_public_estimator_methods)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered public logistic estimator")
@icontract.ensure(lambda result: result == _METHODS, "methods must expose public classification prediction and scoring methods")
def logistic_public_estimator_methods(estimator_name: str) -> tuple[str, ...]:
    """Expose public methods useful for high-level logistic routing."""
    del estimator_name
    return _METHODS


@register_atom(witness_logistic_public_fit_method_payload)
@icontract.require(lambda estimator, X, y: _fit_input_valid(estimator, X, y), "estimator and finite dense fit inputs must satisfy the logistic fit boundary")
@icontract.require(lambda sample_weight, X: _sample_weight_valid(sample_weight, X), "sample_weight must match the sample axis when provided")
@icontract.require(lambda estimator, params: _fit_params_valid(estimator, params), "params must be omitted except for LogisticRegressionCV metadata routing")
@icontract.ensure(lambda result: isinstance(result, dict) and result["method_name"] == "fit", "payload must target public fit")
@icontract.ensure(lambda result, estimator, X, y: _fit_payload_valid(result, estimator, X, y), "fit payload must preserve estimator and positional fit inputs")
def logistic_public_fit_method_payload(
    estimator: object,
    X: object,
    y: object,
    *,
    sample_weight: object = None,
    params: dict[str, object] | None = None,
) -> dict[str, object]:
    """Package a public logistic fit call without running solver or CV work."""
    kwargs: dict[str, object] = {}
    if sample_weight is not None:
        kwargs["sample_weight"] = sample_weight
    if params is not None:
        kwargs.update(params)
    return {"estimator": estimator, "method_name": "fit", "args": (X, y), "kwargs": kwargs}


@register_atom(witness_logistic_public_prediction_method_payload)
@icontract.require(lambda estimator: _fitted_public_logistic(estimator), "estimator must be a fitted public logistic estimator")
@icontract.require(lambda X: _finite_dense_matrix(X), "X must be a finite dense 2D array at the prediction boundary")
@icontract.require(lambda estimator, method_name, X: _method_available(estimator, method_name), "method_name must be available on the fitted estimator")
@icontract.ensure(
    lambda result, estimator, method_name, X: _prediction_payload_valid(result, estimator, method_name, X),
    "prediction payload must preserve estimator, method, positional input, and empty kwargs",
)
def logistic_public_prediction_method_payload(estimator: object, method_name: str, X: object) -> dict[str, object]:
    """Expose a public logistic prediction-like method payload without executing it."""
    return {"estimator": estimator, "method_name": method_name, "args": (X,), "kwargs": {}}


@register_atom(witness_logistic_public_fit_return_self)
@icontract.require(lambda estimator: _fitted_public_logistic(estimator), "estimator must be a fitted public logistic estimator")
@icontract.ensure(lambda result, estimator: result is estimator and _fitted_public_logistic(result), "fit shell must return fitted self")
def logistic_public_fit_return_self(estimator: object) -> object:
    """Return the fitted logistic estimator from the public fit shell."""
    return estimator


@register_atom(witness_logistic_public_fitted_state_summary)
@icontract.require(lambda estimator: _fitted_public_logistic(estimator), "estimator must be a fitted public logistic estimator")
@icontract.ensure(lambda result: isinstance(result, dict) and result["task"] == "classification", "summary must expose classification task metadata")
@icontract.ensure(lambda result: _finite_array(result["coef"]) and _finite_array(result["intercept"]), "summary must expose finite fitted parameters")
def logistic_public_fitted_state_summary(estimator: object) -> dict[str, object]:
    """Expose compact fitted logistic estimator state after solver execution."""
    name = estimator.__class__.__name__
    solver = str(getattr(estimator, "solver", "lbfgs"))
    state: dict[str, object] = {
        "estimator_name": name,
        "task": "classification",
        "family": logistic_public_estimator_family(name),
        "solver_boundary": logistic_public_solver_boundary(name, solver),
        "solver": solver,
        "penalty": getattr(estimator, "penalty", None),
        "classes": tuple(np.asarray(getattr(estimator, "classes_")).tolist()),
        "class_count": int(np.asarray(getattr(estimator, "classes_")).shape[0]),
        "coef": np.asarray(getattr(estimator, "coef_")),
        "intercept": np.asarray(getattr(estimator, "intercept_")),
        "n_features_in": int(getattr(estimator, "n_features_in_")),
        "n_iter": np.asarray(getattr(estimator, "n_iter_")),
        "fit_intercept": bool(getattr(estimator, "fit_intercept")),
    }
    if hasattr(estimator, "C"):
        state["C"] = float(getattr(estimator, "C"))
    if hasattr(estimator, "C_"):
        state["C_"] = np.asarray(getattr(estimator, "C_"))
    if hasattr(estimator, "Cs_"):
        state["Cs_"] = np.asarray(getattr(estimator, "Cs_"))
    if hasattr(estimator, "l1_ratios_"):
        ratios = getattr(estimator, "l1_ratios_")
        state["l1_ratios_"] = None if ratios is None else np.asarray(ratios)
    if name == "LogisticRegressionCV":
        state["cv_boundary"] = "cross_validation_scoring_and_refit"
        state["score_class_count"] = len(getattr(estimator, "scores_", {}))
    return state
