"""Public sklearn decision-tree API-shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_tree_estimator_backend,
    witness_tree_estimator_catalog,
    witness_tree_estimator_family,
    witness_tree_estimator_methods,
    witness_tree_estimator_task,
    witness_tree_fit_method_payload,
    witness_tree_fit_return_self,
    witness_tree_fitted_state_summary,
    witness_tree_prediction_method_payload,
)

_ESTIMATOR_NAMES = ("DecisionTreeClassifier", "DecisionTreeRegressor", "ExtraTreeClassifier", "ExtraTreeRegressor")
_CLASSIFIERS = {"DecisionTreeClassifier", "ExtraTreeClassifier"}
_EXTRA_TREES = {"ExtraTreeClassifier", "ExtraTreeRegressor"}
_BASE_METHODS = {
    "DecisionTreeClassifier": (
        "fit",
        "predict",
        "predict_proba",
        "predict_log_proba",
        "apply",
        "decision_path",
        "score",
        "cost_complexity_pruning_path",
    ),
    "ExtraTreeClassifier": (
        "fit",
        "predict",
        "predict_proba",
        "predict_log_proba",
        "apply",
        "decision_path",
        "score",
        "cost_complexity_pruning_path",
    ),
    "DecisionTreeRegressor": (
        "fit",
        "predict",
        "apply",
        "decision_path",
        "score",
        "cost_complexity_pruning_path",
        "_compute_partial_dependence_recursion",
    ),
    "ExtraTreeRegressor": (
        "fit",
        "predict",
        "apply",
        "decision_path",
        "score",
        "cost_complexity_pruning_path",
    ),
}


def _known_estimator_name(value: object) -> bool:
    return value in _ESTIMATOR_NAMES


def _bool_scalar(value: object) -> bool:
    return isinstance(value, (bool, np.bool_))


def _finite_dense_matrix(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _target_axis_valid(y: object, X: object) -> bool:
    try:
        targets = np.asarray(y)
        matrix = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(matrix.ndim == 2 and targets.ndim in {1, 2} and targets.shape[0] == matrix.shape[0])


def _sample_weight_valid(sample_weight: object, X: object) -> bool:
    if sample_weight is None:
        return True
    try:
        weights = np.asarray(sample_weight, dtype=np.float64)
        matrix = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(weights.ndim == 1 and matrix.ndim == 2 and weights.shape == (matrix.shape[0],) and np.all(np.isfinite(weights)) and np.all(weights >= 0.0))


def _public_tree_estimator(estimator: object) -> bool:
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor

    return isinstance(estimator, (DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor))


def _fitted_public_tree(estimator: object) -> bool:
    return bool(_public_tree_estimator(estimator) and hasattr(estimator, "tree_") and hasattr(estimator, "n_features_in_") and hasattr(estimator, "n_outputs_"))


def _fit_input_valid(estimator: object, X: object, y: object) -> bool:
    return bool(_public_tree_estimator(estimator) and _finite_dense_matrix(X) and _target_axis_valid(y, X))


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


@register_atom(witness_tree_estimator_catalog)
@icontract.require(lambda catalog_scope: catalog_scope == "public_estimators", "catalog_scope must be 'public_estimators'")
@icontract.ensure(lambda result: result == _ESTIMATOR_NAMES, "catalog must expose covered public tree estimators")
def tree_estimator_catalog(catalog_scope: str = "public_estimators") -> tuple[str, ...]:
    """Expose public sklearn.tree estimator names for framework selection."""
    del catalog_scope
    return _ESTIMATOR_NAMES


@register_atom(witness_tree_estimator_family)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered tree estimator")
@icontract.ensure(lambda result: result in {"decision_tree", "extra_tree"}, "family must be decision_tree or extra_tree")
def tree_estimator_family(estimator_name: str) -> str:
    """Return the estimator family for a public sklearn.tree estimator."""
    return "extra_tree" if estimator_name in _EXTRA_TREES else "decision_tree"


@register_atom(witness_tree_estimator_task)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered tree estimator")
@icontract.ensure(lambda result: result in {"classification", "regression"}, "task must be classification or regression")
def tree_estimator_task(estimator_name: str) -> str:
    """Return the high-level learning task for a public sklearn.tree estimator."""
    return "classification" if estimator_name in _CLASSIFIERS else "regression"


@register_atom(witness_tree_estimator_backend)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered tree estimator")
@icontract.ensure(lambda result: result == "cython_tree_builder", "backend boundary must identify the Cython tree builder")
def tree_estimator_backend(estimator_name: str) -> str:
    """Return the native tree-builder boundary behind a public tree estimator."""
    del estimator_name
    return "cython_tree_builder"


@register_atom(witness_tree_estimator_methods)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered tree estimator")
@icontract.ensure(lambda result: isinstance(result, tuple) and "fit" in result and "predict" in result, "methods must include fit and predict")
def tree_estimator_methods(estimator_name: str) -> tuple[str, ...]:
    """Expose public methods useful for high-level tree-estimator routing."""
    return _BASE_METHODS[estimator_name]


@register_atom(witness_tree_fit_method_payload)
@icontract.require(lambda estimator, X, y: _fit_input_valid(estimator, X, y), "estimator and finite dense fit inputs must satisfy the tree fit boundary")
@icontract.require(lambda sample_weight, X: _sample_weight_valid(sample_weight, X), "sample_weight must match the sample axis when provided")
@icontract.require(lambda check_input: _bool_scalar(check_input), "check_input must be boolean")
@icontract.ensure(lambda result: isinstance(result, dict) and result["method_name"] == "fit", "payload must target public fit")
@icontract.ensure(lambda result, estimator, X, y: _fit_payload_valid(result, estimator, X, y), "fit payload must preserve estimator and positional fit inputs")
def tree_fit_method_payload(
    estimator: object,
    X: object,
    y: object,
    *,
    sample_weight: object = None,
    check_input: bool = True,
) -> dict[str, object]:
    """Expose a public tree fit payload without executing native tree building."""
    kwargs: dict[str, object] = {"check_input": bool(check_input)}
    if sample_weight is not None:
        kwargs["sample_weight"] = sample_weight
    return {"estimator": estimator, "method_name": "fit", "args": (X, y), "kwargs": kwargs}


@register_atom(witness_tree_prediction_method_payload)
@icontract.require(lambda estimator: _fitted_public_tree(estimator), "estimator must be a fitted public tree estimator")
@icontract.require(lambda X: _finite_dense_matrix(X), "X must be a finite dense 2D array at the prediction boundary")
@icontract.require(lambda estimator, method_name, X: _method_available(estimator, method_name), "method_name must be available on the fitted estimator")
@icontract.ensure(
    lambda result, estimator, method_name, X: _prediction_payload_valid(result, estimator, method_name, X),
    "prediction payload must preserve estimator, method, positional input, and empty kwargs",
)
def tree_prediction_method_payload(estimator: object, method_name: str, X: object) -> dict[str, object]:
    """Expose a public tree prediction-like method payload without executing it."""
    return {"estimator": estimator, "method_name": method_name, "args": (X,), "kwargs": {}}


@register_atom(witness_tree_fit_return_self)
@icontract.require(lambda estimator: _fitted_public_tree(estimator), "estimator must be a fitted public tree estimator")
@icontract.ensure(lambda result, estimator: result is estimator and _fitted_public_tree(result), "fit shell must return fitted self")
def tree_fit_return_self(estimator: object) -> object:
    """Return the fitted tree estimator from the public fit shell."""
    return estimator


@register_atom(witness_tree_fitted_state_summary)
@icontract.require(lambda estimator: _fitted_public_tree(estimator), "estimator must be a fitted public tree estimator")
@icontract.ensure(lambda result: isinstance(result, dict) and result["backend"] == "cython_tree_builder", "summary must expose the native backend boundary")
@icontract.ensure(lambda result: int(result["node_count"]) >= 1 and int(result["max_depth"]) >= 0, "summary must expose valid tree size metadata")
def tree_fitted_state_summary(estimator: object) -> dict[str, object]:
    """Expose compact fitted-state metadata after native tree construction."""
    name = estimator.__class__.__name__
    tree = getattr(estimator, "tree_")
    state: dict[str, object] = {
        "estimator_name": name,
        "family": tree_estimator_family(name),
        "task": tree_estimator_task(name),
        "backend": tree_estimator_backend(name),
        "n_features_in": int(getattr(estimator, "n_features_in_")),
        "n_outputs": int(getattr(estimator, "n_outputs_")),
        "node_count": int(getattr(tree, "node_count")),
        "max_depth": int(getattr(tree, "max_depth")),
        "n_leaves": int(getattr(tree, "n_leaves")),
    }
    if hasattr(estimator, "classes_"):
        state["classes"] = np.asarray(getattr(estimator, "classes_"))
    if hasattr(estimator, "n_classes_"):
        state["n_classes"] = np.asarray(getattr(estimator, "n_classes_"))
    if hasattr(estimator, "max_features_"):
        state["max_features"] = int(getattr(estimator, "max_features_"))
    return state

