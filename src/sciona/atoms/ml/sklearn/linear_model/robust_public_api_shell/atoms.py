"""Public robust sklearn linear-model API-shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_robust_linear_estimator_boundary,
    witness_robust_linear_estimator_catalog,
    witness_robust_linear_estimator_family,
    witness_robust_linear_estimator_methods,
    witness_robust_linear_fit_method_payload,
    witness_robust_linear_fit_return_self,
    witness_robust_linear_fitted_state_summary,
    witness_robust_linear_prediction_method_payload,
)

_ESTIMATOR_NAMES = ("HuberRegressor", "QuantileRegressor", "RANSACRegressor")
_FAMILIES = {
    "HuberRegressor": "huber",
    "QuantileRegressor": "quantile",
    "RANSACRegressor": "ransac",
}
_BASE_METHODS = {
    "HuberRegressor": ("fit", "predict", "score"),
    "QuantileRegressor": ("fit", "predict", "score"),
    "RANSACRegressor": ("fit", "predict", "score", "get_metadata_routing"),
}
_BOUNDARIES = {"scipy_lbfgs", "scipy_linprog", "estimator_consensus_callbacks"}


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
        targets = np.asarray(y, dtype=np.float64)
        matrix = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(matrix.ndim == 2 and targets.ndim == 1 and targets.shape == (matrix.shape[0],) and np.all(np.isfinite(targets)))


def _sample_weight_valid(sample_weight: object, X: object) -> bool:
    if sample_weight is None:
        return True
    try:
        weights = np.asarray(sample_weight, dtype=np.float64)
        matrix = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(weights.ndim == 1 and matrix.ndim == 2 and weights.shape == (matrix.shape[0],) and np.all(np.isfinite(weights)) and np.all(weights >= 0.0))


def _fit_params_valid(estimator: object, fit_params: object) -> bool:
    if fit_params is None:
        return True
    if estimator.__class__.__name__ != "RANSACRegressor" or not isinstance(fit_params, dict):
        return False
    return all(isinstance(key, str) and key != "" for key in fit_params)


def _public_robust_estimator(estimator: object) -> bool:
    from sklearn.linear_model import HuberRegressor, QuantileRegressor, RANSACRegressor

    return isinstance(estimator, (HuberRegressor, QuantileRegressor, RANSACRegressor))


def _fitted_public_robust(estimator: object) -> bool:
    if not _public_robust_estimator(estimator) or not hasattr(estimator, "n_features_in_"):
        return False
    name = estimator.__class__.__name__
    if name in {"HuberRegressor", "QuantileRegressor"}:
        return bool(hasattr(estimator, "coef_") and hasattr(estimator, "intercept_") and hasattr(estimator, "n_iter_"))
    return bool(hasattr(estimator, "estimator_") and hasattr(estimator, "inlier_mask_") and hasattr(estimator, "n_trials_"))


def _fit_input_valid(estimator: object, X: object, y: object) -> bool:
    return bool(_public_robust_estimator(estimator) and _finite_dense_matrix(X) and _target_vector_valid(y, X))


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


@register_atom(witness_robust_linear_estimator_catalog)
@icontract.require(lambda catalog_scope: catalog_scope == "public_estimators", "catalog_scope must be 'public_estimators'")
@icontract.ensure(lambda result: result == _ESTIMATOR_NAMES, "catalog must expose covered robust estimators")
def robust_linear_estimator_catalog(catalog_scope: str = "public_estimators") -> tuple[str, ...]:
    """Expose public robust linear-model estimator names for selection."""
    del catalog_scope
    return _ESTIMATOR_NAMES


@register_atom(witness_robust_linear_estimator_family)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered robust estimator")
@icontract.ensure(lambda result: result in {"huber", "quantile", "ransac"}, "family must name a covered robust estimator family")
def robust_linear_estimator_family(estimator_name: str) -> str:
    """Return the public robust linear-model estimator family."""
    return _FAMILIES[estimator_name]


@register_atom(witness_robust_linear_estimator_boundary)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered robust estimator")
@icontract.ensure(lambda result: result in _BOUNDARIES, "boundary must name a covered optimizer or callback family")
def robust_linear_estimator_boundary(estimator_name: str) -> str:
    """Return the optimizer or callback boundary for a robust estimator."""
    if estimator_name == "HuberRegressor":
        return "scipy_lbfgs"
    if estimator_name == "QuantileRegressor":
        return "scipy_linprog"
    return "estimator_consensus_callbacks"


@register_atom(witness_robust_linear_estimator_methods)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered robust estimator")
@icontract.ensure(lambda result: isinstance(result, tuple) and "fit" in result and "predict" in result, "methods must include fit and predict")
def robust_linear_estimator_methods(estimator_name: str) -> tuple[str, ...]:
    """Expose public methods useful for high-level robust-model routing."""
    return _BASE_METHODS[estimator_name]


@register_atom(witness_robust_linear_fit_method_payload)
@icontract.require(lambda estimator, X, y: _fit_input_valid(estimator, X, y), "estimator and finite dense fit inputs must satisfy the robust fit boundary")
@icontract.require(lambda sample_weight, X: _sample_weight_valid(sample_weight, X), "sample_weight must match the sample axis when provided")
@icontract.require(lambda estimator, fit_params: _fit_params_valid(estimator, fit_params), "fit_params must be omitted except for RANSAC estimator callback routing")
@icontract.ensure(lambda result: isinstance(result, dict) and result["method_name"] == "fit", "payload must target public fit")
@icontract.ensure(lambda result, estimator, X, y: _fit_payload_valid(result, estimator, X, y), "fit payload must preserve estimator and positional fit inputs")
def robust_linear_fit_method_payload(
    estimator: object,
    X: object,
    y: object,
    *,
    sample_weight: object = None,
    fit_params: dict[str, object] | None = None,
) -> dict[str, object]:
    """Package a public robust linear-model fit call without running it."""
    kwargs: dict[str, object] = {}
    if sample_weight is not None:
        kwargs["sample_weight"] = sample_weight
    if fit_params is not None:
        kwargs.update(fit_params)
    return {"estimator": estimator, "method_name": "fit", "args": (X, y), "kwargs": kwargs}


@register_atom(witness_robust_linear_prediction_method_payload)
@icontract.require(lambda estimator: _fitted_public_robust(estimator), "estimator must be a fitted covered robust estimator")
@icontract.require(lambda X: _finite_dense_matrix(X), "X must be a finite dense 2D array at the prediction boundary")
@icontract.require(lambda estimator, method_name, X: _method_available(estimator, method_name), "method_name must be available on the fitted estimator")
@icontract.ensure(
    lambda result, estimator, method_name, X: _prediction_payload_valid(result, estimator, method_name, X),
    "prediction payload must preserve estimator, method, positional input, and empty kwargs",
)
def robust_linear_prediction_method_payload(estimator: object, method_name: str, X: object) -> dict[str, object]:
    """Expose a robust estimator prediction-like method payload without executing it."""
    return {"estimator": estimator, "method_name": method_name, "args": (X,), "kwargs": {}}


@register_atom(witness_robust_linear_fit_return_self)
@icontract.require(lambda estimator: _fitted_public_robust(estimator), "estimator must be a fitted covered robust estimator")
@icontract.ensure(lambda result, estimator: result is estimator and _fitted_public_robust(result), "fit shell must return fitted self")
def robust_linear_fit_return_self(estimator: object) -> object:
    """Return the fitted robust estimator from the public fit shell."""
    return estimator


@register_atom(witness_robust_linear_fitted_state_summary)
@icontract.require(lambda estimator: _fitted_public_robust(estimator), "estimator must be a fitted covered robust estimator")
@icontract.ensure(lambda result: isinstance(result, dict) and result["boundary"] in _BOUNDARIES, "summary must expose boundary metadata")
def robust_linear_fitted_state_summary(estimator: object) -> dict[str, object]:
    """Expose compact fitted robust-estimator state after external boundaries."""
    name = estimator.__class__.__name__
    state: dict[str, object] = {
        "estimator_name": name,
        "task": "regression",
        "family": robust_linear_estimator_family(name),
        "boundary": robust_linear_estimator_boundary(name),
        "n_features_in": int(getattr(estimator, "n_features_in_")),
    }
    if name in {"HuberRegressor", "QuantileRegressor"}:
        coef = np.asarray(getattr(estimator, "coef_"))
        if not _finite_array(coef):
            raise ValueError("fitted coefficients must be finite")
        state.update(
            {
                "coef": coef,
                "intercept": float(getattr(estimator, "intercept_")),
                "n_iter": int(getattr(estimator, "n_iter_")),
            }
        )
    if name == "HuberRegressor":
        state["scale"] = float(getattr(estimator, "scale_"))
        state["outlier_count"] = int(np.count_nonzero(np.asarray(getattr(estimator, "outliers_"), dtype=bool)))
    if name == "RANSACRegressor":
        inlier_mask = np.asarray(getattr(estimator, "inlier_mask_"), dtype=bool)
        state.update(
            {
                "base_estimator_name": getattr(estimator, "estimator_").__class__.__name__,
                "inlier_count": int(np.count_nonzero(inlier_mask)),
                "sample_count": int(inlier_mask.shape[0]),
                "n_trials": int(getattr(estimator, "n_trials_")),
                "n_skips_no_inliers": int(getattr(estimator, "n_skips_no_inliers_")),
                "n_skips_invalid_data": int(getattr(estimator, "n_skips_invalid_data_")),
                "n_skips_invalid_model": int(getattr(estimator, "n_skips_invalid_model_")),
            }
        )
    return state
