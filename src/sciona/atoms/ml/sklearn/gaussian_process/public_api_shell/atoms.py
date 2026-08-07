"""Public sklearn Gaussian-process API-shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_gp_public_estimator_catalog,
    witness_gp_public_estimator_methods,
    witness_gp_public_estimator_task,
    witness_gp_public_fit_method_payload,
    witness_gp_public_fit_return_self,
    witness_gp_public_fitted_state_summary,
    witness_gp_public_optimizer_boundary,
    witness_gp_public_prediction_method_payload,
)

_ESTIMATOR_NAMES = ("GaussianProcessRegressor", "GaussianProcessClassifier")
_METHODS = {
    "GaussianProcessRegressor": ("fit", "predict", "sample_y", "log_marginal_likelihood", "score"),
    "GaussianProcessClassifier": ("fit", "predict", "predict_proba", "log_marginal_likelihood", "score"),
}
_PREDICTION_METHODS = {
    "GaussianProcessRegressor": {"predict", "sample_y"},
    "GaussianProcessClassifier": {"predict", "predict_proba"},
}
_BOUNDARIES = {"no_optimizer", "scipy_lbfgsb_optimizer", "callable_optimizer"}


def _known_estimator_name(value: object) -> bool:
    return value in _ESTIMATOR_NAMES


def _finite_dense_matrix(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _regression_target_valid(y: object, X: object) -> bool:
    try:
        targets = np.asarray(y, dtype=np.float64)
        matrix = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        matrix.ndim == 2
        and targets.ndim in {1, 2}
        and targets.shape[0] == matrix.shape[0]
        and np.all(np.isfinite(targets))
    )


def _classification_target_valid(y: object, X: object) -> bool:
    try:
        targets = np.asarray(y)
        matrix = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(matrix.ndim == 2 and targets.ndim == 1 and targets.shape == (matrix.shape[0],) and np.unique(targets).shape[0] >= 2)


def _optimizer_valid(optimizer: object) -> bool:
    return optimizer is None or optimizer == "fmin_l_bfgs_b" or callable(optimizer)


def _public_gp_estimator(estimator: object) -> bool:
    from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor

    return isinstance(estimator, (GaussianProcessRegressor, GaussianProcessClassifier))


def _public_gp_regressor(estimator: object) -> bool:
    from sklearn.gaussian_process import GaussianProcessRegressor

    return isinstance(estimator, GaussianProcessRegressor)


def _fit_input_valid(estimator: object, X: object, y: object) -> bool:
    if not _public_gp_estimator(estimator) or not _finite_dense_matrix(X):
        return False
    if _public_gp_regressor(estimator):
        return _regression_target_valid(y, X)
    return _classification_target_valid(y, X)


def _fitted_public_gp(estimator: object) -> bool:
    if not _public_gp_estimator(estimator):
        return False
    common = hasattr(estimator, "kernel_") and hasattr(estimator, "log_marginal_likelihood_value_") and hasattr(estimator, "n_features_in_")
    if _public_gp_regressor(estimator):
        return bool(common and hasattr(estimator, "L_") and hasattr(estimator, "alpha_") and hasattr(estimator, "X_train_"))
    return bool(common and hasattr(estimator, "classes_") and hasattr(estimator, "n_classes_") and hasattr(estimator, "base_estimator_"))


def _method_available(estimator: object, method_name: str) -> bool:
    if not _fitted_public_gp(estimator):
        return False
    name = estimator.__class__.__name__
    return bool(method_name in _PREDICTION_METHODS[name] and hasattr(estimator, method_name))


def _prediction_options_valid(method_name: str, return_std: bool, return_cov: bool, n_samples: int, random_state: object) -> bool:
    if not isinstance(return_std, bool) or not isinstance(return_cov, bool):
        return False
    if method_name == "predict" and return_std and return_cov:
        return False
    if method_name == "sample_y":
        return bool(isinstance(n_samples, int) and not isinstance(n_samples, bool) and n_samples >= 1 and (random_state is None or isinstance(random_state, int)))
    return True


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
        and result.get("kwargs") == {}
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
        and isinstance(result.get("kwargs"), dict)
    )


def _finite_scalar(value: object) -> bool:
    return bool(isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool) and np.isfinite(float(value)))


def _shape_tuple(value: object) -> tuple[int, ...]:
    return tuple(int(dim) for dim in np.asarray(value).shape)


@register_atom(witness_gp_public_estimator_catalog)
@icontract.require(lambda catalog_scope: catalog_scope == "public_estimators", "catalog_scope must be 'public_estimators'")
@icontract.ensure(lambda result: result == _ESTIMATOR_NAMES, "catalog must expose covered public Gaussian-process estimators")
def gp_public_estimator_catalog(catalog_scope: str = "public_estimators") -> tuple[str, ...]:
    """Expose public Gaussian-process estimator names for framework selection."""
    del catalog_scope
    return _ESTIMATOR_NAMES


@register_atom(witness_gp_public_estimator_task)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered public Gaussian-process estimator")
@icontract.ensure(lambda result: result in {"regression", "classification"}, "task must be regression or classification")
def gp_public_estimator_task(estimator_name: str) -> str:
    """Return the public Gaussian-process estimator task family."""
    return "regression" if estimator_name == "GaussianProcessRegressor" else "classification"


@register_atom(witness_gp_public_optimizer_boundary)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered public Gaussian-process estimator")
@icontract.require(lambda optimizer: _optimizer_valid(optimizer), "optimizer must be None, sklearn's L-BFGS-B label, or a callable")
@icontract.ensure(lambda result: result in _BOUNDARIES, "boundary must name a covered optimizer family")
def gp_public_optimizer_boundary(estimator_name: str, optimizer: object) -> str:
    """Return the optimizer boundary selected by a public Gaussian-process estimator."""
    del estimator_name
    if optimizer is None:
        return "no_optimizer"
    if optimizer == "fmin_l_bfgs_b":
        return "scipy_lbfgsb_optimizer"
    return "callable_optimizer"


@register_atom(witness_gp_public_estimator_methods)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered public Gaussian-process estimator")
@icontract.ensure(lambda result: isinstance(result, tuple) and "fit" in result and "predict" in result, "methods must include fit and predict")
def gp_public_estimator_methods(estimator_name: str) -> tuple[str, ...]:
    """Expose public methods useful for high-level Gaussian-process routing."""
    return _METHODS[estimator_name]


@register_atom(witness_gp_public_fit_method_payload)
@icontract.require(lambda estimator, X, y: _fit_input_valid(estimator, X, y), "estimator and finite dense fit inputs must satisfy the Gaussian-process fit boundary")
@icontract.ensure(lambda result: isinstance(result, dict) and result["method_name"] == "fit", "payload must target public fit")
@icontract.ensure(lambda result, estimator, X, y: _fit_payload_valid(result, estimator, X, y), "fit payload must preserve estimator and positional fit inputs")
def gp_public_fit_method_payload(estimator: object, X: object, y: object) -> dict[str, object]:
    """Package a public Gaussian-process fit call without running kernel optimizer or posterior work."""
    return {"estimator": estimator, "method_name": "fit", "args": (X, y), "kwargs": {}}


@register_atom(witness_gp_public_prediction_method_payload)
@icontract.require(lambda estimator: _fitted_public_gp(estimator), "estimator must be a fitted public Gaussian-process estimator")
@icontract.require(lambda X: _finite_dense_matrix(X), "X must be a finite dense 2D array at the prediction boundary")
@icontract.require(lambda estimator, method_name, X: _method_available(estimator, method_name), "method_name must be available on the fitted estimator prediction surface")
@icontract.require(lambda method_name, return_std, return_cov, n_samples, random_state: _prediction_options_valid(method_name, return_std, return_cov, n_samples, random_state), "prediction options must match public Gaussian-process method constraints")
@icontract.ensure(
    lambda result, estimator, method_name, X: _prediction_payload_valid(result, estimator, method_name, X),
    "prediction payload must preserve estimator, method, positional input, and kwargs",
)
def gp_public_prediction_method_payload(
    estimator: object,
    method_name: str,
    X: object,
    *,
    return_std: bool = False,
    return_cov: bool = False,
    n_samples: int = 1,
    random_state: int | None = 0,
) -> dict[str, object]:
    """Expose a public Gaussian-process prediction-like method payload without executing it."""
    kwargs: dict[str, object] = {}
    if method_name == "predict" and _public_gp_regressor(estimator):
        kwargs["return_std"] = return_std
        kwargs["return_cov"] = return_cov
    if method_name == "sample_y":
        kwargs["n_samples"] = n_samples
        kwargs["random_state"] = random_state
    return {"estimator": estimator, "method_name": method_name, "args": (X,), "kwargs": kwargs}


@register_atom(witness_gp_public_fit_return_self)
@icontract.require(lambda estimator: _fitted_public_gp(estimator), "estimator must be a fitted public Gaussian-process estimator")
@icontract.ensure(lambda result, estimator: result is estimator and _fitted_public_gp(result), "fit shell must return fitted self")
def gp_public_fit_return_self(estimator: object) -> object:
    """Return the fitted Gaussian-process estimator from the public fit shell."""
    return estimator


@register_atom(witness_gp_public_fitted_state_summary)
@icontract.require(lambda estimator: _fitted_public_gp(estimator), "estimator must be a fitted public Gaussian-process estimator")
@icontract.ensure(lambda result: isinstance(result, dict) and result["task"] in {"regression", "classification"}, "summary must expose task metadata")
@icontract.ensure(lambda result: result["feature_count"] >= 1 and _finite_scalar(result["log_marginal_likelihood_value"]), "summary must expose fitted feature count and marginal-likelihood value")
def gp_public_fitted_state_summary(estimator: object) -> dict[str, object]:
    """Expose compact fitted Gaussian-process state after delegated sklearn execution."""
    name = estimator.__class__.__name__
    optimizer = getattr(estimator, "optimizer", "fmin_l_bfgs_b")
    state: dict[str, object] = {
        "estimator_name": name,
        "task": gp_public_estimator_task(name),
        "optimizer_boundary": gp_public_optimizer_boundary(name, optimizer),
        "kernel_type": estimator.kernel_.__class__.__name__,
        "kernel_repr": str(estimator.kernel_),
        "feature_count": int(getattr(estimator, "n_features_in_")),
        "log_marginal_likelihood_value": float(getattr(estimator, "log_marginal_likelihood_value_")),
    }
    if name == "GaussianProcessRegressor":
        state["posterior_boundary"] = "cholesky_dual_solve_posterior"
        state["train_shape"] = _shape_tuple(getattr(estimator, "X_train_"))
        state["target_shape"] = _shape_tuple(getattr(estimator, "y_train_"))
        state["L_shape"] = _shape_tuple(getattr(estimator, "L_"))
        state["alpha_shape"] = _shape_tuple(getattr(estimator, "alpha_"))
        state["normalize_y"] = bool(getattr(estimator, "normalize_y"))
        return state
    state["posterior_boundary"] = "laplace_posterior_mode"
    state["multiclass_boundary"] = str(getattr(estimator, "multi_class"))
    state["class_count"] = int(getattr(estimator, "n_classes_"))
    state["classes"] = tuple(np.asarray(getattr(estimator, "classes_")).tolist())
    state["base_estimator_type"] = getattr(estimator, "base_estimator_").__class__.__name__
    if hasattr(getattr(estimator, "base_estimator_"), "X_train_"):
        state["train_shape"] = _shape_tuple(getattr(estimator, "base_estimator_").X_train_)
    return state
