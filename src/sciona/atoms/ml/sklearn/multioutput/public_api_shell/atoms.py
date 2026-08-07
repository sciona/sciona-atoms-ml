"""Public sklearn multioutput meta-estimator API-shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_multioutput_estimator_boundary,
    witness_multioutput_estimator_catalog,
    witness_multioutput_estimator_family,
    witness_multioutput_estimator_methods,
    witness_multioutput_fit_method_payload,
    witness_multioutput_fit_return_self,
    witness_multioutput_fitted_state_summary,
    witness_multioutput_prediction_method_payload,
)

_ESTIMATOR_NAMES = ("MultiOutputClassifier", "MultiOutputRegressor", "ClassifierChain", "RegressorChain")
_FAMILIES = {
    "MultiOutputClassifier": "multioutput_classifier",
    "MultiOutputRegressor": "multioutput_regressor",
    "ClassifierChain": "classifier_chain",
    "RegressorChain": "regressor_chain",
}
_METHODS = {
    "MultiOutputClassifier": ("fit", "partial_fit", "predict", "predict_proba", "score", "get_metadata_routing"),
    "MultiOutputRegressor": ("fit", "partial_fit", "predict", "score", "get_metadata_routing"),
    "ClassifierChain": ("fit", "predict", "predict_proba", "score", "get_metadata_routing"),
    "RegressorChain": ("fit", "predict", "score", "get_metadata_routing"),
}
_BOUNDARIES = {"per_output_estimator_callbacks", "chain_estimator_cv_prediction_callbacks"}


def _known_estimator_name(value: object) -> bool:
    return value in _ESTIMATOR_NAMES


def _finite_dense_matrix(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _target_matrix_valid(y: object, X: object) -> bool:
    try:
        targets = np.asarray(y)
        matrix = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(matrix.ndim == 2 and targets.ndim == 2 and targets.shape[0] == matrix.shape[0] and targets.shape[1] >= 1)


def _sample_weight_valid(sample_weight: object, X: object) -> bool:
    if sample_weight is None:
        return True
    try:
        weights = np.asarray(sample_weight, dtype=np.float64)
        matrix = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(weights.ndim == 1 and matrix.ndim == 2 and weights.shape == (matrix.shape[0],) and np.all(np.isfinite(weights)) and np.all(weights >= 0.0))


def _fit_params_valid(fit_params: object) -> bool:
    if fit_params is None:
        return True
    return bool(isinstance(fit_params, dict) and all(isinstance(key, str) and key != "" for key in fit_params))


def _public_multioutput_estimator(estimator: object) -> bool:
    from sklearn.multioutput import ClassifierChain, MultiOutputClassifier, MultiOutputRegressor, RegressorChain

    return isinstance(estimator, (MultiOutputClassifier, MultiOutputRegressor, ClassifierChain, RegressorChain))


def _fitted_public_multioutput(estimator: object) -> bool:
    return bool(_public_multioutput_estimator(estimator) and hasattr(estimator, "estimators_"))


def _fit_input_valid(estimator: object, X: object, y: object) -> bool:
    return bool(_public_multioutput_estimator(estimator) and _finite_dense_matrix(X) and _target_matrix_valid(y, X))


def _method_available(estimator: object, method_name: str) -> bool:
    return bool(isinstance(method_name, str) and method_name != "" and hasattr(estimator, method_name))


def _fit_payload_valid(result: object, estimator: object, X: object, y: object) -> bool:
    if not isinstance(result, dict):
        return False
    args = result.get("args")
    return bool(
        result.get("estimator") is estimator
        and result.get("method_name") == "fit"
        and isinstance(args, tuple)
        and len(args) == 2
        and args[0] is X
        and args[1] is y
        and isinstance(result.get("kwargs"), dict)
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


@register_atom(witness_multioutput_estimator_catalog)
@icontract.require(lambda catalog_scope: catalog_scope == "public_estimators", "catalog_scope must be 'public_estimators'")
@icontract.ensure(lambda result: result == _ESTIMATOR_NAMES, "catalog must expose covered public multioutput estimators")
def multioutput_estimator_catalog(catalog_scope: str = "public_estimators") -> tuple[str, ...]:
    """Expose public sklearn multioutput meta-estimator names for framework selection."""
    del catalog_scope
    return _ESTIMATOR_NAMES


@register_atom(witness_multioutput_estimator_family)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered public multioutput estimator")
@icontract.ensure(lambda result: result in set(_FAMILIES.values()), "family must name a covered multioutput estimator family")
def multioutput_estimator_family(estimator_name: str) -> str:
    """Return the public multioutput meta-estimator family."""
    return _FAMILIES[estimator_name]


@register_atom(witness_multioutput_estimator_boundary)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered public multioutput estimator")
@icontract.ensure(lambda result: result in _BOUNDARIES, "boundary must name a covered estimator or CV prediction callback family")
def multioutput_estimator_boundary(estimator_name: str) -> str:
    """Return the estimator or chain callback boundary for a multioutput estimator."""
    if estimator_name in {"ClassifierChain", "RegressorChain"}:
        return "chain_estimator_cv_prediction_callbacks"
    return "per_output_estimator_callbacks"


@register_atom(witness_multioutput_estimator_methods)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered public multioutput estimator")
@icontract.ensure(lambda result: isinstance(result, tuple) and "fit" in result and "predict" in result, "methods must include fit and predict")
def multioutput_estimator_methods(estimator_name: str) -> tuple[str, ...]:
    """Expose public methods useful for high-level multioutput routing."""
    return _METHODS[estimator_name]


@register_atom(witness_multioutput_fit_method_payload)
@icontract.require(lambda estimator, X, y: _fit_input_valid(estimator, X, y), "estimator and finite dense fit inputs must satisfy the multioutput fit boundary")
@icontract.require(lambda sample_weight, X: _sample_weight_valid(sample_weight, X), "sample_weight must match the sample axis when provided")
@icontract.require(lambda fit_params: _fit_params_valid(fit_params), "fit_params must be a string-keyed mapping when provided")
@icontract.ensure(lambda result: isinstance(result, dict) and result["method_name"] == "fit", "payload must target public fit")
@icontract.ensure(lambda result, estimator, X, y: _fit_payload_valid(result, estimator, X, y), "fit payload must preserve estimator and positional fit inputs")
def multioutput_fit_method_payload(
    estimator: object,
    X: object,
    y: object,
    *,
    sample_weight: object = None,
    fit_params: dict[str, object] | None = None,
) -> dict[str, object]:
    """Package a public multioutput fit call without executing estimator callbacks."""
    kwargs: dict[str, object] = {}
    if sample_weight is not None:
        kwargs["sample_weight"] = sample_weight
    if fit_params is not None:
        kwargs.update(fit_params)
    return {"estimator": estimator, "method_name": "fit", "args": (X, y), "kwargs": kwargs}


@register_atom(witness_multioutput_prediction_method_payload)
@icontract.require(lambda estimator: _fitted_public_multioutput(estimator), "estimator must be a fitted public multioutput estimator")
@icontract.require(lambda X: _finite_dense_matrix(X), "X must be a finite dense 2D array at the prediction boundary")
@icontract.require(lambda estimator, method_name, X: _method_available(estimator, method_name), "method_name must be available on the fitted estimator")
@icontract.ensure(
    lambda result, estimator, method_name, X: _prediction_payload_valid(result, estimator, method_name, X),
    "prediction payload must preserve estimator, method, positional input, and empty kwargs",
)
def multioutput_prediction_method_payload(estimator: object, method_name: str, X: object) -> dict[str, object]:
    """Expose a public multioutput prediction-like method payload without executing it."""
    return {"estimator": estimator, "method_name": method_name, "args": (X,), "kwargs": {}}


@register_atom(witness_multioutput_fit_return_self)
@icontract.require(lambda estimator: _fitted_public_multioutput(estimator), "estimator must be a fitted public multioutput estimator")
@icontract.ensure(lambda result, estimator: result is estimator and _fitted_public_multioutput(result), "fit shell must return fitted self")
def multioutput_fit_return_self(estimator: object) -> object:
    """Return the fitted multioutput estimator from the public fit shell."""
    return estimator


@register_atom(witness_multioutput_fitted_state_summary)
@icontract.require(lambda estimator: _fitted_public_multioutput(estimator), "estimator must be a fitted public multioutput estimator")
@icontract.ensure(lambda result: isinstance(result, dict) and result["boundary"] in _BOUNDARIES, "summary must expose callback-boundary metadata")
@icontract.ensure(lambda result: result["output_count"] >= 1 and result["estimator_count"] >= 1, "summary must expose fitted outputs and delegates")
def multioutput_fitted_state_summary(estimator: object) -> dict[str, object]:
    """Expose compact fitted multioutput state after delegated callbacks."""
    name = estimator.__class__.__name__
    state: dict[str, object] = {
        "estimator_name": name,
        "family": multioutput_estimator_family(name),
        "boundary": multioutput_estimator_boundary(name),
        "estimator_count": len(getattr(estimator, "estimators_")),
        "output_count": len(getattr(estimator, "estimators_")),
    }
    if hasattr(estimator, "classes_"):
        state["classes_per_output"] = tuple(len(np.asarray(classes)) for classes in getattr(estimator, "classes_"))
    if hasattr(estimator, "order_"):
        state["chain_order"] = tuple(int(index) for index in np.asarray(getattr(estimator, "order_")))
        state["output_count"] = len(state["chain_order"])
    if hasattr(estimator, "n_features_in_"):
        state["n_features_in"] = int(getattr(estimator, "n_features_in_"))
    return state
