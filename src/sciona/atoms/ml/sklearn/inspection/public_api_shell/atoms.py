"""Public sklearn inspection API-shell atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Mapping

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_inspection_public_function_boundary,
    witness_inspection_public_function_catalog,
    witness_partial_dependence_call_payload,
    witness_partial_dependence_result_summary,
    witness_permutation_importance_call_payload,
    witness_permutation_importance_result_summary,
)

_FUNCTION_NAMES = ("partial_dependence", "permutation_importance")
_BOUNDARIES = {"estimator_response_and_grid_callbacks", "scorer_shuffle_and_joblib_callbacks"}
_PD_RESPONSE_METHODS = {"auto", "predict", "predict_proba", "decision_function"}
_PD_METHODS = {"auto", "brute", "recursion"}
_PD_KINDS = {"average", "individual", "both"}


def _known_function_name(value: object) -> bool:
    return value in _FUNCTION_NAMES


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


def _feature_index_valid(value: object, feature_count: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and 0 <= value < feature_count)


def _features_valid(features: object, X: object) -> bool:
    try:
        matrix = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if matrix.ndim != 2:
        return False
    feature_count = int(matrix.shape[1])
    if _feature_index_valid(features, feature_count):
        return True
    if not isinstance(features, (list, tuple)) or len(features) < 1:
        return False
    for item in features:
        if _feature_index_valid(item, feature_count):
            continue
        if isinstance(item, tuple) and len(item) >= 1 and all(_feature_index_valid(index, feature_count) for index in item):
            continue
        return False
    return True


def _percentiles_valid(percentiles: object) -> bool:
    if not isinstance(percentiles, tuple) or len(percentiles) != 2:
        return False
    low, high = percentiles
    return bool(
        isinstance(low, (int, float))
        and isinstance(high, (int, float))
        and not isinstance(low, bool)
        and not isinstance(high, bool)
        and 0.0 <= float(low) < float(high) <= 1.0
    )


def _grid_resolution_valid(grid_resolution: object) -> bool:
    return bool(isinstance(grid_resolution, int) and not isinstance(grid_resolution, bool) and grid_resolution >= 1)


def _pd_options_valid(response_method: object, method: object, kind: object) -> bool:
    return bool(response_method in _PD_RESPONSE_METHODS and method in _PD_METHODS and kind in _PD_KINDS)


def _random_state_valid(random_state: object) -> bool:
    return random_state is None or (isinstance(random_state, int) and not isinstance(random_state, bool))


def _n_jobs_valid(n_jobs: object) -> bool:
    return n_jobs is None or (isinstance(n_jobs, int) and not isinstance(n_jobs, bool) and n_jobs != 0)


def _n_repeats_valid(n_repeats: object) -> bool:
    return bool(isinstance(n_repeats, int) and not isinstance(n_repeats, bool) and n_repeats >= 1)


def _max_samples_valid(max_samples: object, X: object) -> bool:
    try:
        matrix = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if matrix.ndim != 2:
        return False
    sample_count = int(matrix.shape[0])
    if isinstance(max_samples, int) and not isinstance(max_samples, bool):
        return 1 <= max_samples <= sample_count
    if isinstance(max_samples, float) and not isinstance(max_samples, bool):
        return 0.0 < max_samples <= 1.0
    return False


def _scoring_valid(scoring: object) -> bool:
    if scoring is None or isinstance(scoring, str) or callable(scoring):
        return True
    if isinstance(scoring, (list, tuple)):
        return bool(len(scoring) >= 1 and all(isinstance(name, str) and name != "" for name in scoring))
    if isinstance(scoring, Mapping):
        return bool(len(scoring) >= 1 and all(isinstance(name, str) and name != "" for name in scoring.keys()))
    return False


def _estimator_response_capable(estimator: object) -> bool:
    return bool(hasattr(estimator, "predict") or hasattr(estimator, "predict_proba") or hasattr(estimator, "decision_function"))


def _partial_payload_valid(result: object, estimator: object, X: object, features: object) -> bool:
    if not isinstance(result, dict):
        return False
    args = result.get("args")
    kwargs = result.get("kwargs")
    return bool(
        result.get("function_name") == "partial_dependence"
        and isinstance(args, tuple)
        and len(args) == 3
        and args[0] is estimator
        and args[1] is X
        and args[2] is features
        and isinstance(kwargs, dict)
    )


def _permutation_payload_valid(result: object, estimator: object, X: object, y: object) -> bool:
    if not isinstance(result, dict):
        return False
    args = result.get("args")
    kwargs = result.get("kwargs")
    return bool(
        result.get("function_name") == "permutation_importance"
        and isinstance(args, tuple)
        and len(args) == 3
        and args[0] is estimator
        and args[1] is X
        and args[2] is y
        and isinstance(kwargs, dict)
    )


def _array_shape(value: object) -> tuple[int, ...]:
    return tuple(int(dim) for dim in np.asarray(value).shape)


def _partial_result_valid(result: object) -> bool:
    return bool(hasattr(result, "keys") and "grid_values" in result and ("average" in result or "individual" in result))


def _permutation_bunch_valid(value: object) -> bool:
    return bool(hasattr(value, "keys") and {"importances_mean", "importances_std", "importances"}.issubset(value.keys()))


def _permutation_result_valid(result: object) -> bool:
    if _permutation_bunch_valid(result):
        return True
    return bool(isinstance(result, dict) and len(result) >= 1 and all(_permutation_bunch_valid(value) for value in result.values()))


@register_atom(witness_inspection_public_function_catalog)
@icontract.require(lambda catalog_scope: catalog_scope == "public_functions", "catalog_scope must be 'public_functions'")
@icontract.ensure(lambda result: result == _FUNCTION_NAMES, "catalog must expose covered public inspection functions")
def inspection_public_function_catalog(catalog_scope: str = "public_functions") -> tuple[str, ...]:
    """Expose public inspection function names for framework selection."""
    del catalog_scope
    return _FUNCTION_NAMES


@register_atom(witness_inspection_public_function_boundary)
@icontract.require(lambda function_name: _known_function_name(function_name), "function_name must name a covered public inspection function")
@icontract.ensure(lambda result: result in _BOUNDARIES, "boundary must name a covered inspection callback family")
def inspection_public_function_boundary(function_name: str) -> str:
    """Return the callback boundary represented by a public inspection function."""
    if function_name == "partial_dependence":
        return "estimator_response_and_grid_callbacks"
    return "scorer_shuffle_and_joblib_callbacks"


@register_atom(witness_partial_dependence_call_payload)
@icontract.require(lambda estimator: _estimator_response_capable(estimator), "estimator must expose a prediction-like response method")
@icontract.require(lambda X: _finite_dense_matrix(X), "X must be a finite dense 2D array")
@icontract.require(lambda features, X: _features_valid(features, X), "features must identify valid columns")
@icontract.require(lambda sample_weight, X: _sample_weight_valid(sample_weight, X), "sample_weight must match the sample axis when provided")
@icontract.require(lambda percentiles: _percentiles_valid(percentiles), "percentiles must be an increasing pair in [0, 1]")
@icontract.require(lambda grid_resolution: _grid_resolution_valid(grid_resolution), "grid_resolution must be positive")
@icontract.require(lambda response_method, method, kind: _pd_options_valid(response_method, method, kind), "partial_dependence options must be public sklearn options")
@icontract.ensure(lambda result: isinstance(result, dict) and result["function_name"] == "partial_dependence", "payload must target partial_dependence")
@icontract.ensure(lambda result, estimator, X, features: _partial_payload_valid(result, estimator, X, features), "payload must preserve positional partial_dependence inputs")
def partial_dependence_call_payload(
    estimator: object,
    X: object,
    features: object,
    *,
    sample_weight: object = None,
    categorical_features: object = None,
    feature_names: object = None,
    response_method: str = "auto",
    percentiles: tuple[float, float] = (0.05, 0.95),
    grid_resolution: int = 100,
    custom_values: object = None,
    method: str = "auto",
    kind: str = "average",
) -> dict[str, object]:
    """Package a public partial_dependence call without executing estimator responses."""
    kwargs = {
        "sample_weight": sample_weight,
        "categorical_features": categorical_features,
        "feature_names": feature_names,
        "response_method": response_method,
        "percentiles": percentiles,
        "grid_resolution": grid_resolution,
        "custom_values": custom_values,
        "method": method,
        "kind": kind,
    }
    return {"function_name": "partial_dependence", "args": (estimator, X, features), "kwargs": kwargs}


@register_atom(witness_permutation_importance_call_payload)
@icontract.require(lambda estimator: hasattr(estimator, "score") or hasattr(estimator, "predict"), "estimator must expose score or prediction behavior")
@icontract.require(lambda X: _finite_dense_matrix(X), "X must be a finite dense 2D array")
@icontract.require(lambda y, X: _target_axis_valid(y, X), "y must match the sample axis")
@icontract.require(lambda scoring: _scoring_valid(scoring), "scoring must be None, a scorer, or named scorer collection")
@icontract.require(lambda n_repeats: _n_repeats_valid(n_repeats), "n_repeats must be positive")
@icontract.require(lambda n_jobs: _n_jobs_valid(n_jobs), "n_jobs must be None or nonzero integer")
@icontract.require(lambda random_state: _random_state_valid(random_state), "random_state must be None or an integer seed")
@icontract.require(lambda sample_weight, X: _sample_weight_valid(sample_weight, X), "sample_weight must match the sample axis when provided")
@icontract.require(lambda max_samples, X: _max_samples_valid(max_samples, X), "max_samples must be a valid public subsample size")
@icontract.ensure(lambda result: isinstance(result, dict) and result["function_name"] == "permutation_importance", "payload must target permutation_importance")
@icontract.ensure(lambda result, estimator, X, y: _permutation_payload_valid(result, estimator, X, y), "payload must preserve positional permutation_importance inputs")
def permutation_importance_call_payload(
    estimator: object,
    X: object,
    y: object,
    *,
    scoring: object = None,
    n_repeats: int = 5,
    n_jobs: int | None = None,
    random_state: int | None = None,
    sample_weight: object = None,
    max_samples: int | float = 1.0,
) -> dict[str, object]:
    """Package a public permutation_importance call without executing scorer or shuffle work."""
    kwargs = {
        "scoring": scoring,
        "n_repeats": n_repeats,
        "n_jobs": n_jobs,
        "random_state": random_state,
        "sample_weight": sample_weight,
        "max_samples": max_samples,
    }
    return {"function_name": "permutation_importance", "args": (estimator, X, y), "kwargs": kwargs}


@register_atom(witness_partial_dependence_result_summary)
@icontract.require(lambda inspection_result: _partial_result_valid(inspection_result), "inspection_result must look like sklearn's partial_dependence Bunch")
@icontract.ensure(lambda result: isinstance(result, dict) and result["function_name"] == "partial_dependence", "summary must identify partial_dependence")
@icontract.ensure(lambda result: result["grid_value_count"] >= 1 and ("average_shape" in result or "individual_shape" in result), "summary must expose grid and prediction shapes")
def partial_dependence_result_summary(inspection_result: object) -> dict[str, object]:
    """Expose compact public partial_dependence result metadata."""
    summary: dict[str, object] = {
        "function_name": "partial_dependence",
        "boundary": inspection_public_function_boundary("partial_dependence"),
        "grid_value_count": len(inspection_result["grid_values"]),
        "grid_value_lengths": tuple(int(np.asarray(values).shape[0]) for values in inspection_result["grid_values"]),
    }
    if "average" in inspection_result:
        summary["average_shape"] = _array_shape(inspection_result["average"])
    if "individual" in inspection_result:
        summary["individual_shape"] = _array_shape(inspection_result["individual"])
    return summary


@register_atom(witness_permutation_importance_result_summary)
@icontract.require(lambda inspection_result: _permutation_result_valid(inspection_result), "inspection_result must look like sklearn's permutation_importance Bunch or metric dict")
@icontract.ensure(lambda result: isinstance(result, dict) and result["function_name"] == "permutation_importance", "summary must identify permutation_importance")
@icontract.ensure(lambda result: result["metric_count"] >= 1 and result["feature_count"] >= 1, "summary must expose metric and feature counts")
def permutation_importance_result_summary(inspection_result: object) -> dict[str, object]:
    """Expose compact public permutation_importance result metadata."""
    if _permutation_bunch_valid(inspection_result):
        return {
            "function_name": "permutation_importance",
            "boundary": inspection_public_function_boundary("permutation_importance"),
            "metric_count": 1,
            "metric_names": ("default",),
            "feature_count": int(np.asarray(inspection_result["importances_mean"]).shape[0]),
            "importances_shape": _array_shape(inspection_result["importances"]),
        }
    metric_names = tuple(str(name) for name in inspection_result.keys())
    first = inspection_result[metric_names[0]]
    return {
        "function_name": "permutation_importance",
        "boundary": inspection_public_function_boundary("permutation_importance"),
        "metric_count": len(metric_names),
        "metric_names": metric_names,
        "feature_count": int(np.asarray(first["importances_mean"]).shape[0]),
        "importances_shape": _array_shape(first["importances"]),
    }
