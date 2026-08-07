"""Public sklearn GLM regressor API-shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_glm_estimator_catalog,
    witness_glm_estimator_distribution,
    witness_glm_estimator_methods,
    witness_glm_estimator_optimizer,
    witness_glm_fit_method_payload,
    witness_glm_fit_return_self,
    witness_glm_fitted_state_summary,
    witness_glm_prediction_method_payload,
)

_ESTIMATOR_NAMES = ("PoissonRegressor", "GammaRegressor", "TweedieRegressor")
_BASE_METHODS = ("fit", "predict", "score")
_SOLVER_NAMES = {"lbfgs", "newton-cholesky"}


def _known_estimator_name(value: object) -> bool:
    return value in _ESTIMATOR_NAMES


def _finite_dense_matrix(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _target_vector_valid(y: object, X: object, estimator: object) -> bool:
    try:
        targets = np.asarray(y, dtype=np.float64)
        matrix = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if not (matrix.ndim == 2 and targets.ndim == 1 and targets.shape == (matrix.shape[0],) and np.all(np.isfinite(targets))):
        return False
    name = estimator.__class__.__name__
    if name == "GammaRegressor":
        return bool(np.all(targets > 0.0))
    if name == "PoissonRegressor":
        return bool(np.all(targets >= 0.0))
    power = float(getattr(estimator, "power", 0.0))
    if power > 0.0:
        return bool(np.all(targets >= 0.0))
    return True


def _sample_weight_valid(sample_weight: object, X: object) -> bool:
    if sample_weight is None:
        return True
    try:
        weights = np.asarray(sample_weight, dtype=np.float64)
        matrix = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(weights.ndim == 1 and matrix.ndim == 2 and weights.shape == (matrix.shape[0],) and np.all(np.isfinite(weights)) and np.all(weights >= 0.0))


def _public_glm_estimator(estimator: object) -> bool:
    from sklearn.linear_model import GammaRegressor, PoissonRegressor, TweedieRegressor

    return isinstance(estimator, (PoissonRegressor, GammaRegressor, TweedieRegressor))


def _fitted_public_glm(estimator: object) -> bool:
    return bool(
        _public_glm_estimator(estimator)
        and hasattr(estimator, "coef_")
        and hasattr(estimator, "intercept_")
        and hasattr(estimator, "n_features_in_")
        and hasattr(estimator, "n_iter_")
    )


def _fit_input_valid(estimator: object, X: object, y: object) -> bool:
    return bool(_public_glm_estimator(estimator) and _finite_dense_matrix(X) and _target_vector_valid(y, X, estimator))


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


@register_atom(witness_glm_estimator_catalog)
@icontract.require(lambda catalog_scope: catalog_scope == "public_estimators", "catalog_scope must be 'public_estimators'")
@icontract.ensure(lambda result: result == _ESTIMATOR_NAMES, "catalog must expose covered public GLM estimators")
def glm_estimator_catalog(catalog_scope: str = "public_estimators") -> tuple[str, ...]:
    """Expose public sklearn GLM regressor names for framework selection."""
    del catalog_scope
    return _ESTIMATOR_NAMES


@register_atom(witness_glm_estimator_distribution)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered GLM estimator")
@icontract.ensure(lambda result: result in {"poisson", "gamma", "tweedie"}, "distribution must be a covered GLM distribution")
def glm_estimator_distribution(estimator_name: str, power: float | None = None) -> str:
    """Return the response distribution family for a public GLM estimator."""
    del power
    if estimator_name == "PoissonRegressor":
        return "poisson"
    if estimator_name == "GammaRegressor":
        return "gamma"
    return "tweedie"


@register_atom(witness_glm_estimator_optimizer)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered GLM estimator")
@icontract.require(lambda solver: solver in _SOLVER_NAMES, "solver must be an exposed GLM optimizer boundary")
@icontract.ensure(lambda result: result in _SOLVER_NAMES, "optimizer boundary must be an exposed GLM solver")
def glm_estimator_optimizer(estimator_name: str, solver: str) -> str:
    """Return the optimizer boundary selected for a public GLM estimator."""
    del estimator_name
    return solver


@register_atom(witness_glm_estimator_methods)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered GLM estimator")
@icontract.ensure(lambda result: result == _BASE_METHODS, "methods must expose fit, predict, and score")
def glm_estimator_methods(estimator_name: str) -> tuple[str, ...]:
    """Expose public methods useful for high-level GLM routing."""
    del estimator_name
    return _BASE_METHODS


@register_atom(witness_glm_fit_method_payload)
@icontract.require(lambda estimator, X, y: _fit_input_valid(estimator, X, y), "estimator and finite dense fit inputs must satisfy the GLM fit boundary")
@icontract.require(lambda sample_weight, X: _sample_weight_valid(sample_weight, X), "sample_weight must match the sample axis when provided")
@icontract.ensure(lambda result: isinstance(result, dict) and result["method_name"] == "fit", "payload must target public fit")
@icontract.ensure(lambda result, estimator, X, y: _fit_payload_valid(result, estimator, X, y), "fit payload must preserve estimator and positional fit inputs")
def glm_fit_method_payload(
    estimator: object,
    X: object,
    y: object,
    *,
    sample_weight: object = None,
) -> dict[str, object]:
    """Package a public generalized linear model fit call without running it."""
    kwargs = {} if sample_weight is None else {"sample_weight": sample_weight}
    return {"estimator": estimator, "method_name": "fit", "args": (X, y), "kwargs": kwargs}


@register_atom(witness_glm_prediction_method_payload)
@icontract.require(lambda estimator: _fitted_public_glm(estimator), "estimator must be a fitted public GLM estimator")
@icontract.require(lambda X: _finite_dense_matrix(X), "X must be a finite dense 2D array at the prediction boundary")
@icontract.require(lambda estimator, method_name, X: _method_available(estimator, method_name), "method_name must be available on the fitted estimator")
@icontract.ensure(
    lambda result, estimator, method_name, X: _prediction_payload_valid(result, estimator, method_name, X),
    "prediction payload must preserve estimator, method, positional input, and empty kwargs",
)
def glm_prediction_method_payload(estimator: object, method_name: str, X: object) -> dict[str, object]:
    """Expose a public GLM prediction-like method payload without executing it."""
    return {"estimator": estimator, "method_name": method_name, "args": (X,), "kwargs": {}}


@register_atom(witness_glm_fit_return_self)
@icontract.require(lambda estimator: _fitted_public_glm(estimator), "estimator must be a fitted public GLM estimator")
@icontract.ensure(lambda result, estimator: result is estimator and _fitted_public_glm(result), "fit shell must return fitted self")
def glm_fit_return_self(estimator: object) -> object:
    """Return the fitted GLM estimator from the public fit shell."""
    return estimator


@register_atom(witness_glm_fitted_state_summary)
@icontract.require(lambda estimator: _fitted_public_glm(estimator), "estimator must be a fitted public GLM estimator")
@icontract.ensure(lambda result: isinstance(result, dict) and result["task"] == "regression", "summary must expose regression task metadata")
@icontract.ensure(lambda result: _finite_array(result["coef"]) and np.isfinite(float(result["intercept"])), "summary must expose finite coefficients")
def glm_fitted_state_summary(estimator: object) -> dict[str, object]:
    """Expose compact fitted GLM coefficient state after optimizer execution."""
    name = estimator.__class__.__name__
    solver = str(getattr(estimator, "solver", "lbfgs"))
    state: dict[str, object] = {
        "estimator_name": name,
        "task": "regression",
        "distribution": glm_estimator_distribution(name, float(getattr(estimator, "power", 0.0)) if hasattr(estimator, "power") else None),
        "optimizer": glm_estimator_optimizer(name, solver),
        "coef": np.asarray(getattr(estimator, "coef_")),
        "intercept": float(getattr(estimator, "intercept_")),
        "n_features_in": int(getattr(estimator, "n_features_in_")),
        "n_iter": int(getattr(estimator, "n_iter_")),
        "alpha": float(getattr(estimator, "alpha")),
        "fit_intercept": bool(getattr(estimator, "fit_intercept")),
    }
    if hasattr(estimator, "power"):
        state["power"] = float(getattr(estimator, "power"))
    return state
