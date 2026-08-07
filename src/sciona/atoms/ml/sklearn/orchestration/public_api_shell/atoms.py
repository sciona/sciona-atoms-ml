"""Public sklearn orchestration API-shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_orchestration_boundary,
    witness_orchestration_estimator_catalog,
    witness_orchestration_estimator_family,
    witness_orchestration_estimator_methods,
    witness_orchestration_fit_method_payload,
    witness_orchestration_fit_return_self,
    witness_orchestration_fitted_state_summary,
    witness_orchestration_method_payload,
)

_ESTIMATOR_NAMES = ("Pipeline", "ColumnTransformer", "FeatureUnion", "GridSearchCV", "RandomizedSearchCV")
_FAMILIES = {
    "Pipeline": "pipeline",
    "ColumnTransformer": "column_transformer",
    "FeatureUnion": "feature_union",
    "GridSearchCV": "grid_search_cv",
    "RandomizedSearchCV": "randomized_search_cv",
}
_METHODS = {
    "Pipeline": ("fit", "transform", "fit_transform", "predict", "score", "get_params", "set_params"),
    "ColumnTransformer": ("fit", "transform", "fit_transform", "get_params", "set_params"),
    "FeatureUnion": ("fit", "transform", "fit_transform", "get_params", "set_params"),
    "GridSearchCV": ("fit", "predict", "score", "transform", "get_params", "set_params"),
    "RandomizedSearchCV": ("fit", "predict", "score", "transform", "get_params", "set_params"),
}
_BOUNDARIES = {
    "pipeline_step_callbacks",
    "column_transformer_callbacks",
    "parallel_transformer_callbacks",
    "cv_search_fit_score_callbacks",
}


def _known_estimator_name(value: object) -> bool:
    return value in _ESTIMATOR_NAMES


def _finite_dense_matrix(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _target_axis_valid(y: object, X: object) -> bool:
    if y is None:
        return True
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


def _public_orchestration_estimator(estimator: object) -> bool:
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.pipeline import FeatureUnion, Pipeline

    return isinstance(estimator, (Pipeline, ColumnTransformer, FeatureUnion, GridSearchCV, RandomizedSearchCV))


def _fitted_public_orchestration(estimator: object) -> bool:
    if not _public_orchestration_estimator(estimator):
        return False
    name = estimator.__class__.__name__
    if name in {"GridSearchCV", "RandomizedSearchCV"}:
        return bool(hasattr(estimator, "best_estimator_") and hasattr(estimator, "cv_results_"))
    if name == "ColumnTransformer":
        return bool(hasattr(estimator, "transformers_"))
    if name == "FeatureUnion":
        return any(_component_has_fitted_state(transformer) for _, transformer in getattr(estimator, "transformer_list", ()))
    if name == "Pipeline":
        return any(_component_has_fitted_state(step) for _, step in getattr(estimator, "steps", ()))
    return False


def _component_has_fitted_state(component: object) -> bool:
    if component in {"drop", "passthrough"} or component is None:
        return False
    return any(
        hasattr(component, attribute)
        for attribute in ("n_features_in_", "classes_", "coef_", "mean_", "scale_", "components_", "statistics_", "tree_")
    )


def _fit_input_valid(estimator: object, X: object, y: object) -> bool:
    return bool(_public_orchestration_estimator(estimator) and _finite_dense_matrix(X) and _target_axis_valid(y, X))


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


def _method_payload_valid(result: object, estimator: object, method_name: str, X: object) -> bool:
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


def _named_items(items: object) -> tuple[str, ...]:
    try:
        return tuple(str(item[0]) for item in items)
    except (TypeError, ValueError):
        return ()


@register_atom(witness_orchestration_estimator_catalog)
@icontract.require(lambda catalog_scope: catalog_scope == "public_estimators", "catalog_scope must be 'public_estimators'")
@icontract.ensure(lambda result: result == _ESTIMATOR_NAMES, "catalog must expose covered public orchestration estimators")
def orchestration_estimator_catalog(catalog_scope: str = "public_estimators") -> tuple[str, ...]:
    """Expose public sklearn orchestration estimator names for framework selection."""
    del catalog_scope
    return _ESTIMATOR_NAMES


@register_atom(witness_orchestration_estimator_family)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered public orchestration estimator")
@icontract.ensure(lambda result: result in set(_FAMILIES.values()), "family must name a covered public orchestration family")
def orchestration_estimator_family(estimator_name: str) -> str:
    """Return the public sklearn orchestration estimator family."""
    return _FAMILIES[estimator_name]


@register_atom(witness_orchestration_boundary)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered public orchestration estimator")
@icontract.ensure(lambda result: result in _BOUNDARIES, "boundary must name a covered callback family")
def orchestration_boundary(estimator_name: str) -> str:
    """Return the callback boundary behind a public orchestration estimator."""
    if estimator_name == "ColumnTransformer":
        return "column_transformer_callbacks"
    if estimator_name == "FeatureUnion":
        return "parallel_transformer_callbacks"
    if estimator_name in {"GridSearchCV", "RandomizedSearchCV"}:
        return "cv_search_fit_score_callbacks"
    return "pipeline_step_callbacks"


@register_atom(witness_orchestration_estimator_methods)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered public orchestration estimator")
@icontract.ensure(lambda result: isinstance(result, tuple) and "fit" in result, "methods must include fit")
def orchestration_estimator_methods(estimator_name: str) -> tuple[str, ...]:
    """Expose public methods useful for high-level orchestration routing."""
    return _METHODS[estimator_name]


@register_atom(witness_orchestration_fit_method_payload)
@icontract.require(lambda estimator, X, y: _fit_input_valid(estimator, X, y), "estimator and finite dense fit inputs must satisfy the orchestration fit boundary")
@icontract.require(lambda params: _params_valid(params), "params must be a string-keyed mapping when provided")
@icontract.ensure(lambda result: isinstance(result, dict) and result["method_name"] == "fit", "payload must target public fit")
@icontract.ensure(lambda result, estimator, X, y: _fit_payload_valid(result, estimator, X, y), "fit payload must preserve estimator and positional fit inputs")
def orchestration_fit_method_payload(
    estimator: object,
    X: object,
    y: object = None,
    *,
    params: dict[str, object] | None = None,
) -> dict[str, object]:
    """Package a public orchestration fit call without executing callbacks."""
    args = (X,) if y is None else (X, y)
    kwargs = {} if params is None else dict(params)
    return {"estimator": estimator, "method_name": "fit", "args": args, "kwargs": kwargs}


@register_atom(witness_orchestration_method_payload)
@icontract.require(lambda estimator: _fitted_public_orchestration(estimator), "estimator must be a fitted public orchestration estimator")
@icontract.require(lambda X: _finite_dense_matrix(X), "X must be a finite dense 2D array at the method boundary")
@icontract.require(lambda estimator, method_name, X: _method_available(estimator, method_name), "method_name must be available on the fitted estimator")
@icontract.ensure(
    lambda result, estimator, method_name, X: _method_payload_valid(result, estimator, method_name, X),
    "method payload must preserve estimator, method, positional input, and empty kwargs",
)
def orchestration_method_payload(estimator: object, method_name: str, X: object) -> dict[str, object]:
    """Expose a public orchestration transform or prediction payload without executing it."""
    return {"estimator": estimator, "method_name": method_name, "args": (X,), "kwargs": {}}


@register_atom(witness_orchestration_fit_return_self)
@icontract.require(lambda estimator: _fitted_public_orchestration(estimator), "estimator must be a fitted public orchestration estimator")
@icontract.ensure(lambda result, estimator: result is estimator and _fitted_public_orchestration(result), "fit shell must return fitted self")
def orchestration_fit_return_self(estimator: object) -> object:
    """Return the fitted orchestration estimator from the public fit shell."""
    return estimator


@register_atom(witness_orchestration_fitted_state_summary)
@icontract.require(lambda estimator: _fitted_public_orchestration(estimator), "estimator must be a fitted public orchestration estimator")
@icontract.ensure(lambda result: isinstance(result, dict) and result["boundary"] in _BOUNDARIES, "summary must expose boundary metadata")
def orchestration_fitted_state_summary(estimator: object) -> dict[str, object]:
    """Expose compact fitted orchestration state after delegated callbacks."""
    name = estimator.__class__.__name__
    state: dict[str, object] = {
        "estimator_name": name,
        "family": orchestration_estimator_family(name),
        "boundary": orchestration_boundary(name),
    }
    if name == "Pipeline":
        state["step_names"] = _named_items(getattr(estimator, "steps", ()))
        state["step_count"] = len(getattr(estimator, "steps", ()))
    if name == "ColumnTransformer":
        state["transformer_names"] = _named_items(getattr(estimator, "transformers_", ()))
        state["transformer_count"] = len(getattr(estimator, "transformers_", ()))
    if name == "FeatureUnion":
        state["transformer_names"] = _named_items(getattr(estimator, "transformer_list", ()))
        state["transformer_count"] = len(getattr(estimator, "transformer_list", ()))
    if name in {"GridSearchCV", "RandomizedSearchCV"}:
        state.update(
            {
                "best_estimator_name": getattr(estimator, "best_estimator_").__class__.__name__,
                "best_index": int(getattr(estimator, "best_index_")),
                "best_score": float(getattr(estimator, "best_score_")),
                "candidate_count": len(getattr(estimator, "cv_results_", {}).get("params", ())),
                "refit": bool(getattr(estimator, "refit")),
            }
        )
    if hasattr(estimator, "n_features_in_"):
        state["n_features_in"] = int(getattr(estimator, "n_features_in_"))
    return state
