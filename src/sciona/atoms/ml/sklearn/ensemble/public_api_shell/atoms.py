"""Public sklearn ensemble API-shell atoms adapted from scikit-learn."""

from __future__ import annotations

from typing import Any

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_ensemble_estimator_backend,
    witness_ensemble_estimator_catalog,
    witness_ensemble_estimator_family,
    witness_ensemble_estimator_methods,
    witness_ensemble_estimator_task,
    witness_ensemble_fit_return_self,
    witness_ensemble_fitted_state_summary,
    witness_ensemble_prediction_method_payload,
)

_ESTIMATOR_NAMES = (
    "RandomForestClassifier",
    "RandomForestRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor",
    "IsolationForest",
)
_CLASSIFIERS = {"RandomForestClassifier", "ExtraTreesClassifier", "GradientBoostingClassifier", "HistGradientBoostingClassifier"}
_REGRESSORS = {"RandomForestRegressor", "ExtraTreesRegressor", "GradientBoostingRegressor", "HistGradientBoostingRegressor"}
_FORESTS = {"RandomForestClassifier", "RandomForestRegressor"}
_EXTRA_TREES = {"ExtraTreesClassifier", "ExtraTreesRegressor"}
_GRADIENT_BOOSTING = {"GradientBoostingClassifier", "GradientBoostingRegressor"}
_HIST_GRADIENT_BOOSTING = {"HistGradientBoostingClassifier", "HistGradientBoostingRegressor"}
_BASE_METHODS = {
    "RandomForestClassifier": ("fit", "predict", "predict_proba", "predict_log_proba", "apply", "decision_path", "score"),
    "ExtraTreesClassifier": ("fit", "predict", "predict_proba", "predict_log_proba", "apply", "decision_path", "score"),
    "GradientBoostingClassifier": ("fit", "predict", "predict_proba", "predict_log_proba", "decision_function", "apply", "staged_predict", "score"),
    "HistGradientBoostingClassifier": ("fit", "predict", "predict_proba", "decision_function", "staged_predict", "score"),
    "RandomForestRegressor": ("fit", "predict", "apply", "decision_path", "score"),
    "ExtraTreesRegressor": ("fit", "predict", "apply", "decision_path", "score"),
    "GradientBoostingRegressor": ("fit", "predict", "apply", "staged_predict", "score"),
    "HistGradientBoostingRegressor": ("fit", "predict", "staged_predict", "score"),
    "IsolationForest": ("fit", "fit_predict", "predict", "decision_function", "score_samples"),
}


def _known_estimator_name(value: object) -> bool:
    return value in _ESTIMATOR_NAMES


def _bool_scalar(value: object) -> bool:
    return isinstance(value, (bool, np.bool_))


def _fitted_public_ensemble(estimator: object) -> bool:
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        ExtraTreesRegressor,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
        IsolationForest,
        RandomForestClassifier,
        RandomForestRegressor,
    )

    if isinstance(estimator, (RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor, IsolationForest)):
        return bool(hasattr(estimator, "estimators_") and hasattr(estimator, "n_features_in_"))
    if isinstance(estimator, (GradientBoostingClassifier, GradientBoostingRegressor)):
        return bool(hasattr(estimator, "estimators_") and hasattr(estimator, "n_estimators_") and hasattr(estimator, "n_features_in_"))
    if isinstance(estimator, (HistGradientBoostingClassifier, HistGradientBoostingRegressor)):
        return bool(hasattr(estimator, "_predictors") and hasattr(estimator, "n_iter_") and hasattr(estimator, "n_features_in_"))
    return False


def _method_available(estimator: object, method_name: str) -> bool:
    return bool(isinstance(method_name, str) and method_name != "" and hasattr(estimator, method_name))


def _payload_result_valid(result: object, estimator: object, method_name: str, X: object) -> bool:
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


def _estimator_count(estimator: object) -> int:
    if hasattr(estimator, "estimators_"):
        return int(np.asarray(getattr(estimator, "estimators_"), dtype=object).size)
    predictors = getattr(estimator, "_predictors")
    return int(len(predictors))


@register_atom(witness_ensemble_estimator_catalog)
@icontract.require(lambda catalog_scope: catalog_scope == "public_estimators", "catalog_scope must be 'public_estimators'")
@icontract.ensure(lambda result: result == _ESTIMATOR_NAMES, "catalog must expose covered public ensemble estimators")
def ensemble_estimator_catalog(
    catalog_scope: str = "public_estimators",
) -> tuple[str, ...]:
    """Expose public tree-ensemble estimator names for framework selection."""
    del catalog_scope
    return _ESTIMATOR_NAMES


@register_atom(witness_ensemble_estimator_family)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered ensemble estimator")
@icontract.ensure(
    lambda result: result in {"random_forest", "extra_trees", "gradient_boosting", "hist_gradient_boosting", "isolation_forest"},
    "family must be one of the covered ensemble families",
)
def ensemble_estimator_family(estimator_name: str) -> str:
    """Return the ensemble family for a public tree-ensemble estimator."""
    if estimator_name in _FORESTS:
        return "random_forest"
    if estimator_name in _EXTRA_TREES:
        return "extra_trees"
    if estimator_name in _GRADIENT_BOOSTING:
        return "gradient_boosting"
    if estimator_name in _HIST_GRADIENT_BOOSTING:
        return "hist_gradient_boosting"
    return "isolation_forest"


@register_atom(witness_ensemble_estimator_task)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered ensemble estimator")
@icontract.ensure(
    lambda result: result in {"classification", "regression", "outlier_detection"},
    "task must be one of the covered high-level ensemble tasks",
)
def ensemble_estimator_task(estimator_name: str) -> str:
    """Return the high-level learning task for a public tree-ensemble estimator."""
    if estimator_name in _CLASSIFIERS:
        return "classification"
    if estimator_name in _REGRESSORS:
        return "regression"
    return "outlier_detection"


@register_atom(witness_ensemble_estimator_backend)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered ensemble estimator")
@icontract.ensure(
    lambda result: result in {"cython_tree_ensemble", "python_tree_boosting", "histogram_boosting", "isolation_tree_ensemble"},
    "backend boundary must name the tree/boosting execution family",
)
def ensemble_estimator_backend(estimator_name: str) -> str:
    """Return the native or solver boundary family behind a public tree ensemble."""
    if estimator_name in _FORESTS or estimator_name in _EXTRA_TREES:
        return "cython_tree_ensemble"
    if estimator_name in _GRADIENT_BOOSTING:
        return "python_tree_boosting"
    if estimator_name in _HIST_GRADIENT_BOOSTING:
        return "histogram_boosting"
    return "isolation_tree_ensemble"


@register_atom(witness_ensemble_estimator_methods)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered ensemble estimator")
@icontract.ensure(lambda result: isinstance(result, tuple) and "fit" in result and "predict" in result, "methods must include fit and predict")
def ensemble_estimator_methods(estimator_name: str) -> tuple[str, ...]:
    """Expose public methods useful for high-level tree-ensemble routing."""
    return _BASE_METHODS[estimator_name]


@register_atom(witness_ensemble_prediction_method_payload)
@icontract.require(lambda estimator: _fitted_public_ensemble(estimator), "estimator must be a fitted covered public ensemble estimator")
@icontract.require(lambda estimator, method_name: _method_available(estimator, method_name), "method_name must be available on the fitted estimator")
@icontract.ensure(
    lambda result, estimator, method_name, X: _payload_result_valid(result, estimator, method_name, X),
    "prediction payload must preserve estimator, method, positional input, and empty kwargs",
)
def ensemble_prediction_method_payload(
    estimator: object,
    method_name: str,
    X: object,
) -> dict[str, object]:
    """Expose a public ensemble prediction-like method payload without executing it."""
    return {"estimator": estimator, "method_name": method_name, "args": (X,), "kwargs": {}}


@register_atom(witness_ensemble_fit_return_self)
@icontract.require(lambda estimator: _fitted_public_ensemble(estimator), "estimator must be a fitted covered public ensemble estimator")
@icontract.ensure(lambda result, estimator: result is estimator and _fitted_public_ensemble(result), "fit shell must return fitted self")
def ensemble_fit_return_self(estimator: object) -> object:
    """Return the fitted ensemble estimator from the public fit shell."""
    return estimator


@register_atom(witness_ensemble_fitted_state_summary)
@icontract.require(lambda estimator: _fitted_public_ensemble(estimator), "estimator must be a fitted covered public ensemble estimator")
@icontract.ensure(lambda result: isinstance(result, dict) and int(result["estimator_count"]) >= 1, "summary must expose at least one fitted base learner")
def ensemble_fitted_state_summary(estimator: object) -> dict[str, Any]:
    """Expose a compact fitted-state summary after deferred tree/boosting work."""
    name = estimator.__class__.__name__
    state: dict[str, Any] = {
        "estimator_name": name,
        "family": ensemble_estimator_family(name),
        "task": ensemble_estimator_task(name),
        "backend": ensemble_estimator_backend(name),
        "n_features_in": int(getattr(estimator, "n_features_in_")),
        "estimator_count": _estimator_count(estimator),
    }
    if hasattr(estimator, "classes_"):
        state["classes"] = np.asarray(getattr(estimator, "classes_"))
    if hasattr(estimator, "n_classes_"):
        state["n_classes"] = np.asarray(getattr(estimator, "n_classes_"))
    if hasattr(estimator, "n_outputs_"):
        state["n_outputs"] = int(getattr(estimator, "n_outputs_"))
    if hasattr(estimator, "n_estimators_"):
        state["n_estimators"] = int(getattr(estimator, "n_estimators_"))
    if hasattr(estimator, "n_iter_"):
        state["n_iter"] = int(getattr(estimator, "n_iter_"))
    if hasattr(estimator, "offset_"):
        state["offset"] = float(getattr(estimator, "offset_"))
    return state
