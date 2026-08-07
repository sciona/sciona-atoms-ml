"""Public sklearn multiclass meta-estimator API-shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_multiclass_estimator_boundary,
    witness_multiclass_estimator_catalog,
    witness_multiclass_estimator_family,
    witness_multiclass_estimator_methods,
    witness_multiclass_fit_method_payload,
    witness_multiclass_fit_return_self,
    witness_multiclass_fitted_state_summary,
    witness_multiclass_prediction_method_payload,
)

_ESTIMATOR_NAMES = ("OneVsRestClassifier", "OneVsOneClassifier", "OutputCodeClassifier")
_FAMILIES = {
    "OneVsRestClassifier": "one_vs_rest",
    "OneVsOneClassifier": "one_vs_one",
    "OutputCodeClassifier": "output_code",
}
_METHODS = {
    "OneVsRestClassifier": ("fit", "partial_fit", "predict", "predict_proba", "decision_function", "score", "get_metadata_routing"),
    "OneVsOneClassifier": ("fit", "partial_fit", "predict", "decision_function", "score", "get_metadata_routing"),
    "OutputCodeClassifier": ("fit", "predict", "score", "get_metadata_routing"),
}
_BOUNDARIES = {"cloned_estimator_response_callbacks", "code_book_estimator_response_callbacks"}


def _known_estimator_name(value: object) -> bool:
    return value in _ESTIMATOR_NAMES


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
    if not (matrix.ndim == 2 and targets.ndim in {1, 2} and targets.shape[0] == matrix.shape[0]):
        return False
    if targets.ndim == 1:
        return bool(np.unique(targets).shape[0] >= 2)
    return bool(targets.shape[1] >= 1)


def _fit_params_valid(fit_params: object) -> bool:
    if fit_params is None:
        return True
    return bool(isinstance(fit_params, dict) and all(isinstance(key, str) and key != "" for key in fit_params))


def _public_multiclass_estimator(estimator: object) -> bool:
    from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier, OutputCodeClassifier

    return isinstance(estimator, (OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier))


def _fitted_public_multiclass(estimator: object) -> bool:
    return bool(_public_multiclass_estimator(estimator) and hasattr(estimator, "classes_") and hasattr(estimator, "estimators_"))


def _fit_input_valid(estimator: object, X: object, y: object) -> bool:
    return bool(_public_multiclass_estimator(estimator) and _finite_dense_matrix(X) and _target_vector_valid(y, X))


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


@register_atom(witness_multiclass_estimator_catalog)
@icontract.require(lambda catalog_scope: catalog_scope == "public_estimators", "catalog_scope must be 'public_estimators'")
@icontract.ensure(lambda result: result == _ESTIMATOR_NAMES, "catalog must expose covered public multiclass estimators")
def multiclass_estimator_catalog(catalog_scope: str = "public_estimators") -> tuple[str, ...]:
    """Expose public sklearn multiclass meta-estimator names for framework selection."""
    del catalog_scope
    return _ESTIMATOR_NAMES


@register_atom(witness_multiclass_estimator_family)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered public multiclass estimator")
@icontract.ensure(lambda result: result in set(_FAMILIES.values()), "family must name a covered multiclass estimator family")
def multiclass_estimator_family(estimator_name: str) -> str:
    """Return the public multiclass meta-estimator family."""
    return _FAMILIES[estimator_name]


@register_atom(witness_multiclass_estimator_boundary)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered public multiclass estimator")
@icontract.ensure(lambda result: result in _BOUNDARIES, "boundary must name a covered estimator-response callback family")
def multiclass_estimator_boundary(estimator_name: str) -> str:
    """Return the cloned-estimator response boundary for a multiclass estimator."""
    if estimator_name == "OutputCodeClassifier":
        return "code_book_estimator_response_callbacks"
    return "cloned_estimator_response_callbacks"


@register_atom(witness_multiclass_estimator_methods)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered public multiclass estimator")
@icontract.ensure(lambda result: isinstance(result, tuple) and "fit" in result and "predict" in result, "methods must include fit and predict")
def multiclass_estimator_methods(estimator_name: str) -> tuple[str, ...]:
    """Expose public methods useful for high-level multiclass routing."""
    return _METHODS[estimator_name]


@register_atom(witness_multiclass_fit_method_payload)
@icontract.require(lambda estimator, X, y: _fit_input_valid(estimator, X, y), "estimator and finite dense fit inputs must satisfy the multiclass fit boundary")
@icontract.require(lambda fit_params: _fit_params_valid(fit_params), "fit_params must be a string-keyed mapping when provided")
@icontract.ensure(lambda result: isinstance(result, dict) and result["method_name"] == "fit", "payload must target public fit")
@icontract.ensure(lambda result, estimator, X, y: _fit_payload_valid(result, estimator, X, y), "fit payload must preserve estimator and positional fit inputs")
def multiclass_fit_method_payload(
    estimator: object,
    X: object,
    y: object,
    *,
    fit_params: dict[str, object] | None = None,
) -> dict[str, object]:
    """Package a public multiclass fit call without executing estimator callbacks."""
    kwargs = {} if fit_params is None else dict(fit_params)
    return {"estimator": estimator, "method_name": "fit", "args": (X, y), "kwargs": kwargs}


@register_atom(witness_multiclass_prediction_method_payload)
@icontract.require(lambda estimator: _fitted_public_multiclass(estimator), "estimator must be a fitted public multiclass estimator")
@icontract.require(lambda X: _finite_dense_matrix(X), "X must be a finite dense 2D array at the prediction boundary")
@icontract.require(lambda estimator, method_name, X: _method_available(estimator, method_name), "method_name must be available on the fitted estimator")
@icontract.ensure(
    lambda result, estimator, method_name, X: _prediction_payload_valid(result, estimator, method_name, X),
    "prediction payload must preserve estimator, method, positional input, and empty kwargs",
)
def multiclass_prediction_method_payload(estimator: object, method_name: str, X: object) -> dict[str, object]:
    """Expose a public multiclass prediction-like method payload without executing it."""
    return {"estimator": estimator, "method_name": method_name, "args": (X,), "kwargs": {}}


@register_atom(witness_multiclass_fit_return_self)
@icontract.require(lambda estimator: _fitted_public_multiclass(estimator), "estimator must be a fitted public multiclass estimator")
@icontract.ensure(lambda result, estimator: result is estimator and _fitted_public_multiclass(result), "fit shell must return fitted self")
def multiclass_fit_return_self(estimator: object) -> object:
    """Return the fitted multiclass estimator from the public fit shell."""
    return estimator


@register_atom(witness_multiclass_fitted_state_summary)
@icontract.require(lambda estimator: _fitted_public_multiclass(estimator), "estimator must be a fitted public multiclass estimator")
@icontract.ensure(lambda result: isinstance(result, dict) and result["boundary"] in _BOUNDARIES, "summary must expose callback-boundary metadata")
@icontract.ensure(lambda result: result["class_count"] >= 2 and result["estimator_count"] >= 1, "summary must expose fitted classes and delegates")
def multiclass_fitted_state_summary(estimator: object) -> dict[str, object]:
    """Expose compact fitted multiclass state after delegated callbacks."""
    name = estimator.__class__.__name__
    classes = np.asarray(getattr(estimator, "classes_"))
    state: dict[str, object] = {
        "estimator_name": name,
        "family": multiclass_estimator_family(name),
        "boundary": multiclass_estimator_boundary(name),
        "classes": tuple(classes.tolist()),
        "class_count": int(classes.shape[0]),
        "estimator_count": len(getattr(estimator, "estimators_")),
    }
    if hasattr(estimator, "n_features_in_"):
        state["n_features_in"] = int(getattr(estimator, "n_features_in_"))
    if hasattr(estimator, "label_binarizer_"):
        state["multilabel"] = bool(getattr(estimator.label_binarizer_, "y_type_", "") == "multilabel-indicator")
    if hasattr(estimator, "pairwise_indices_"):
        pairwise_indices = getattr(estimator, "pairwise_indices_")
        state["pairwise_indices_available"] = pairwise_indices is not None
    if hasattr(estimator, "code_book_"):
        code_book = np.asarray(getattr(estimator, "code_book_"))
        state["code_book_shape"] = tuple(int(axis) for axis in code_book.shape)
    return state
