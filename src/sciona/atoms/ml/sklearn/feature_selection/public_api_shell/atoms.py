"""Public sklearn feature-selection selector API-shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_feature_selector_boundary,
    witness_feature_selector_catalog,
    witness_feature_selector_family,
    witness_feature_selector_fit_method_payload,
    witness_feature_selector_fit_return_self,
    witness_feature_selector_fitted_state_summary,
    witness_feature_selector_methods,
    witness_feature_selector_transform_method_payload,
)

_ESTIMATOR_NAMES = ("RFE", "RFECV", "SelectFromModel", "SequentialFeatureSelector")
_FAMILIES = {
    "RFE": "recursive_feature_elimination",
    "RFECV": "recursive_feature_elimination_cv",
    "SelectFromModel": "select_from_model",
    "SequentialFeatureSelector": "sequential_feature_selector",
}
_METHODS = {
    "RFE": ("fit", "transform", "fit_transform", "inverse_transform", "get_support", "predict", "score"),
    "RFECV": ("fit", "transform", "fit_transform", "inverse_transform", "get_support", "predict", "score"),
    "SelectFromModel": ("fit", "transform", "fit_transform", "inverse_transform", "get_support"),
    "SequentialFeatureSelector": ("fit", "transform", "fit_transform", "inverse_transform", "get_support"),
}
_BOUNDARIES = {
    "estimator_importance_callbacks",
    "estimator_importance_cv_scorer_callbacks",
    "estimator_cv_scorer_callbacks",
}


def _known_estimator_name(value: object) -> bool:
    return value in _ESTIMATOR_NAMES


def _finite_dense_matrix(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _target_axis_valid(estimator: object, y: object, X: object) -> bool:
    name = estimator.__class__.__name__
    if y is None:
        return name in {"SelectFromModel", "SequentialFeatureSelector"}
    try:
        targets = np.asarray(y)
        matrix = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(matrix.ndim == 2 and targets.ndim in {1, 2} and targets.shape[0] == matrix.shape[0])


def _params_valid(params: object) -> bool:
    if params is None:
        return True
    return bool(isinstance(params, dict) and all(isinstance(key, str) and key != "" for key in params))


def _public_feature_selector(estimator: object) -> bool:
    from sklearn.feature_selection import RFE, RFECV, SelectFromModel, SequentialFeatureSelector

    return isinstance(estimator, (RFE, RFECV, SelectFromModel, SequentialFeatureSelector))


def _fitted_public_feature_selector(estimator: object) -> bool:
    return bool(_public_feature_selector(estimator) and hasattr(estimator, "n_features_in_") and (hasattr(estimator, "support_") or hasattr(estimator, "estimator_")))


def _fit_input_valid(estimator: object, X: object, y: object) -> bool:
    return bool(_public_feature_selector(estimator) and _finite_dense_matrix(X) and _target_axis_valid(estimator, y, X))


def _method_available(estimator: object, method_name: str) -> bool:
    return bool(isinstance(method_name, str) and method_name != "" and hasattr(estimator, method_name))


def _fit_payload_valid(result: object, estimator: object, X: object, y: object) -> bool:
    if not isinstance(result, dict):
        return False
    args = result.get("args")
    expected_len = 1 if y is None else 2
    return bool(
        result.get("estimator") is estimator
        and result.get("method_name") == "fit"
        and isinstance(args, tuple)
        and len(args) == expected_len
        and args[0] is X
        and (expected_len == 1 or args[1] is y)
        and isinstance(result.get("kwargs"), dict)
    )


def _transform_payload_valid(result: object, estimator: object, method_name: str, X: object) -> bool:
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


@register_atom(witness_feature_selector_catalog)
@icontract.require(lambda catalog_scope: catalog_scope == "public_estimators", "catalog_scope must be 'public_estimators'")
@icontract.ensure(lambda result: result == _ESTIMATOR_NAMES, "catalog must expose covered public feature selectors")
def feature_selector_catalog(catalog_scope: str = "public_estimators") -> tuple[str, ...]:
    """Expose public sklearn feature-selector names for framework selection."""
    del catalog_scope
    return _ESTIMATOR_NAMES


@register_atom(witness_feature_selector_family)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered public feature selector")
@icontract.ensure(lambda result: result in set(_FAMILIES.values()), "family must name a covered feature selector family")
def feature_selector_family(estimator_name: str) -> str:
    """Return the public feature-selector family."""
    return _FAMILIES[estimator_name]


@register_atom(witness_feature_selector_boundary)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered public feature selector")
@icontract.ensure(lambda result: result in _BOUNDARIES, "boundary must name a covered callback family")
def feature_selector_boundary(estimator_name: str) -> str:
    """Return the estimator, importance, scorer, or CV callback boundary."""
    if estimator_name == "RFECV":
        return "estimator_importance_cv_scorer_callbacks"
    if estimator_name == "SequentialFeatureSelector":
        return "estimator_cv_scorer_callbacks"
    return "estimator_importance_callbacks"


@register_atom(witness_feature_selector_methods)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered public feature selector")
@icontract.ensure(lambda result: isinstance(result, tuple) and "fit" in result and "transform" in result, "methods must include fit and transform")
def feature_selector_methods(estimator_name: str) -> tuple[str, ...]:
    """Expose public methods useful for high-level feature-selector routing."""
    return _METHODS[estimator_name]


@register_atom(witness_feature_selector_fit_method_payload)
@icontract.require(lambda estimator, X, y: _fit_input_valid(estimator, X, y), "estimator and finite dense fit inputs must satisfy the selector fit boundary")
@icontract.require(lambda params: _params_valid(params), "params must be a string-keyed mapping when provided")
@icontract.ensure(lambda result: isinstance(result, dict) and result["method_name"] == "fit", "payload must target public fit")
@icontract.ensure(lambda result, estimator, X, y: _fit_payload_valid(result, estimator, X, y), "fit payload must preserve estimator and positional fit inputs")
def feature_selector_fit_method_payload(
    estimator: object,
    X: object,
    y: object = None,
    *,
    params: dict[str, object] | None = None,
) -> dict[str, object]:
    """Package a public feature-selector fit call without executing callbacks."""
    args = (X,) if y is None else (X, y)
    kwargs = {} if params is None else dict(params)
    return {"estimator": estimator, "method_name": "fit", "args": args, "kwargs": kwargs}


@register_atom(witness_feature_selector_transform_method_payload)
@icontract.require(lambda estimator: _fitted_public_feature_selector(estimator), "estimator must be a fitted public feature selector")
@icontract.require(lambda X: _finite_dense_matrix(X), "X must be a finite dense 2D array at the selector method boundary")
@icontract.require(lambda estimator, method_name, X: _method_available(estimator, method_name), "method_name must be available on the fitted selector")
@icontract.ensure(
    lambda result, estimator, method_name, X: _transform_payload_valid(result, estimator, method_name, X),
    "method payload must preserve estimator, method, positional input, and empty kwargs",
)
def feature_selector_transform_method_payload(estimator: object, method_name: str, X: object) -> dict[str, object]:
    """Expose a public selector transform-like method payload without executing it."""
    return {"estimator": estimator, "method_name": method_name, "args": (X,), "kwargs": {}}


@register_atom(witness_feature_selector_fit_return_self)
@icontract.require(lambda estimator: _fitted_public_feature_selector(estimator), "estimator must be a fitted public feature selector")
@icontract.ensure(lambda result, estimator: result is estimator and _fitted_public_feature_selector(result), "fit shell must return fitted self")
def feature_selector_fit_return_self(estimator: object) -> object:
    """Return the fitted feature selector from the public fit shell."""
    return estimator


@register_atom(witness_feature_selector_fitted_state_summary)
@icontract.require(lambda estimator: _fitted_public_feature_selector(estimator), "estimator must be a fitted public feature selector")
@icontract.ensure(lambda result: isinstance(result, dict) and result["boundary"] in _BOUNDARIES, "summary must expose callback-boundary metadata")
@icontract.ensure(lambda result: result["selected_feature_count"] >= 1, "summary must include at least one selected feature")
def feature_selector_fitted_state_summary(estimator: object) -> dict[str, object]:
    """Expose compact fitted selector state after delegated callbacks."""
    name = estimator.__class__.__name__
    support = np.asarray(getattr(estimator, "support_", estimator.get_support()), dtype=bool)
    state: dict[str, object] = {
        "estimator_name": name,
        "family": feature_selector_family(name),
        "boundary": feature_selector_boundary(name),
        "n_features_in": int(getattr(estimator, "n_features_in_")),
        "selected_feature_count": int(np.count_nonzero(support)),
        "support": support,
    }
    if hasattr(estimator, "ranking_"):
        state["ranking"] = np.asarray(getattr(estimator, "ranking_"))
    if hasattr(estimator, "estimator_"):
        state["delegate_estimator_name"] = getattr(estimator, "estimator_").__class__.__name__
    if hasattr(estimator, "cv_results_"):
        state["cv_candidate_count"] = len(getattr(estimator, "cv_results_", {}).get("n_features", ()))
    if hasattr(estimator, "n_features_to_select_"):
        state["n_features_to_select"] = int(getattr(estimator, "n_features_to_select_"))
    return state
