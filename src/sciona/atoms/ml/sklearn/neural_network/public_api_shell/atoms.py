"""Public sklearn MLP estimator API-shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_mlp_public_estimator_catalog,
    witness_mlp_public_estimator_methods,
    witness_mlp_public_estimator_task,
    witness_mlp_public_fit_method_payload,
    witness_mlp_public_fit_return_self,
    witness_mlp_public_fitted_state_summary,
    witness_mlp_public_optimizer_boundary,
    witness_mlp_public_prediction_method_payload,
)

_ESTIMATOR_NAMES = ("MLPClassifier", "MLPRegressor")
_METHODS = {
    "MLPClassifier": ("fit", "partial_fit", "predict", "predict_proba", "predict_log_proba", "score"),
    "MLPRegressor": ("fit", "partial_fit", "predict", "score"),
}
_PREDICTION_METHODS = {
    "MLPClassifier": {"predict", "predict_proba", "predict_log_proba"},
    "MLPRegressor": {"predict"},
}
_SOLVERS = {"lbfgs", "sgd", "adam"}
_BOUNDARIES = {"scipy_lbfgs_optimizer", "stochastic_sgd_optimizer", "stochastic_adam_optimizer"}


def _known_estimator_name(value: object) -> bool:
    return value in _ESTIMATOR_NAMES


def _known_solver(value: object) -> bool:
    return value in _SOLVERS


def _finite_dense_matrix(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _classifier_target_valid(y: object, X: object) -> bool:
    try:
        targets = np.asarray(y)
        matrix = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if matrix.ndim != 2 or targets.shape[0] != matrix.shape[0]:
        return False
    if targets.ndim == 1:
        return bool(np.unique(targets).shape[0] >= 2)
    return bool(targets.ndim == 2 and targets.shape[1] >= 1)


def _regressor_target_valid(y: object, X: object) -> bool:
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


def _sample_weight_valid(sample_weight: object, X: object) -> bool:
    if sample_weight is None:
        return True
    try:
        weights = np.asarray(sample_weight, dtype=np.float64)
        matrix = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(weights.ndim == 1 and matrix.ndim == 2 and weights.shape == (matrix.shape[0],) and np.all(np.isfinite(weights)) and np.all(weights >= 0.0))


def _public_mlp_estimator(estimator: object) -> bool:
    from sklearn.neural_network import MLPClassifier, MLPRegressor

    return isinstance(estimator, (MLPClassifier, MLPRegressor))


def _public_mlp_classifier(estimator: object) -> bool:
    from sklearn.neural_network import MLPClassifier

    return isinstance(estimator, MLPClassifier)


def _fit_input_valid(estimator: object, X: object, y: object) -> bool:
    if not _public_mlp_estimator(estimator) or not _finite_dense_matrix(X):
        return False
    if _public_mlp_classifier(estimator):
        return _classifier_target_valid(y, X)
    return _regressor_target_valid(y, X)


def _fitted_public_mlp(estimator: object) -> bool:
    return bool(
        _public_mlp_estimator(estimator)
        and hasattr(estimator, "coefs_")
        and hasattr(estimator, "intercepts_")
        and hasattr(estimator, "n_iter_")
        and hasattr(estimator, "n_layers_")
        and hasattr(estimator, "n_outputs_")
        and hasattr(estimator, "out_activation_")
    )


def _method_available(estimator: object, method_name: str) -> bool:
    if not _fitted_public_mlp(estimator):
        return False
    name = estimator.__class__.__name__
    return bool(method_name in _PREDICTION_METHODS[name] and hasattr(estimator, method_name))


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


@register_atom(witness_mlp_public_estimator_catalog)
@icontract.require(lambda catalog_scope: catalog_scope == "public_estimators", "catalog_scope must be 'public_estimators'")
@icontract.ensure(lambda result: result == _ESTIMATOR_NAMES, "catalog must expose covered public MLP estimators")
def mlp_public_estimator_catalog(catalog_scope: str = "public_estimators") -> tuple[str, ...]:
    """Expose public MLP estimator names for framework selection."""
    del catalog_scope
    return _ESTIMATOR_NAMES


@register_atom(witness_mlp_public_estimator_task)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered public MLP estimator")
@icontract.ensure(lambda result: result in {"classification", "regression"}, "task must be classification or regression")
def mlp_public_estimator_task(estimator_name: str) -> str:
    """Return the public MLP estimator task family."""
    return "classification" if estimator_name == "MLPClassifier" else "regression"


@register_atom(witness_mlp_public_optimizer_boundary)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered public MLP estimator")
@icontract.require(lambda solver: _known_solver(solver), "solver must be an exposed MLP solver")
@icontract.ensure(lambda result: result in _BOUNDARIES, "boundary must name a covered MLP optimizer family")
def mlp_public_optimizer_boundary(estimator_name: str, solver: str) -> str:
    """Return the optimizer boundary selected by a public MLP estimator."""
    del estimator_name
    if solver == "lbfgs":
        return "scipy_lbfgs_optimizer"
    if solver == "sgd":
        return "stochastic_sgd_optimizer"
    return "stochastic_adam_optimizer"


@register_atom(witness_mlp_public_estimator_methods)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered public MLP estimator")
@icontract.ensure(lambda result: isinstance(result, tuple) and "fit" in result and "predict" in result, "methods must include fit and predict")
def mlp_public_estimator_methods(estimator_name: str) -> tuple[str, ...]:
    """Expose public methods useful for high-level MLP routing."""
    return _METHODS[estimator_name]


@register_atom(witness_mlp_public_fit_method_payload)
@icontract.require(lambda estimator, X, y: _fit_input_valid(estimator, X, y), "estimator and finite dense fit inputs must satisfy the MLP fit boundary")
@icontract.require(lambda sample_weight, X: _sample_weight_valid(sample_weight, X), "sample_weight must match the sample axis when provided")
@icontract.ensure(lambda result: isinstance(result, dict) and result["method_name"] == "fit", "payload must target public fit")
@icontract.ensure(lambda result, estimator, X, y: _fit_payload_valid(result, estimator, X, y), "fit payload must preserve estimator and positional fit inputs")
def mlp_public_fit_method_payload(
    estimator: object,
    X: object,
    y: object,
    *,
    sample_weight: object = None,
) -> dict[str, object]:
    """Package a public MLP fit call without running optimizer or backpropagation work."""
    kwargs: dict[str, object] = {}
    if sample_weight is not None:
        kwargs["sample_weight"] = sample_weight
    return {"estimator": estimator, "method_name": "fit", "args": (X, y), "kwargs": kwargs}


@register_atom(witness_mlp_public_prediction_method_payload)
@icontract.require(lambda estimator: _fitted_public_mlp(estimator), "estimator must be a fitted public MLP estimator")
@icontract.require(lambda X: _finite_dense_matrix(X), "X must be a finite dense 2D array at the prediction boundary")
@icontract.require(lambda estimator, method_name, X: _method_available(estimator, method_name), "method_name must be available on the fitted estimator prediction surface")
@icontract.ensure(
    lambda result, estimator, method_name, X: _prediction_payload_valid(result, estimator, method_name, X),
    "prediction payload must preserve estimator, method, positional input, and empty kwargs",
)
def mlp_public_prediction_method_payload(estimator: object, method_name: str, X: object) -> dict[str, object]:
    """Expose a public MLP prediction-like method payload without executing it."""
    return {"estimator": estimator, "method_name": method_name, "args": (X,), "kwargs": {}}


@register_atom(witness_mlp_public_fit_return_self)
@icontract.require(lambda estimator: _fitted_public_mlp(estimator), "estimator must be a fitted public MLP estimator")
@icontract.ensure(lambda result, estimator: result is estimator and _fitted_public_mlp(result), "fit shell must return fitted self")
def mlp_public_fit_return_self(estimator: object) -> object:
    """Return the fitted MLP estimator from the public fit shell."""
    return estimator


@register_atom(witness_mlp_public_fitted_state_summary)
@icontract.require(lambda estimator: _fitted_public_mlp(estimator), "estimator must be a fitted public MLP estimator")
@icontract.ensure(lambda result: isinstance(result, dict) and result["task"] in {"classification", "regression"}, "summary must expose task metadata")
@icontract.ensure(lambda result: result["layer_count"] >= 2 and result["output_count"] >= 1, "summary must expose fitted layer and output counts")
def mlp_public_fitted_state_summary(estimator: object) -> dict[str, object]:
    """Expose compact fitted MLP state after sklearn optimizer execution."""
    name = estimator.__class__.__name__
    solver = str(getattr(estimator, "solver", "adam"))
    state: dict[str, object] = {
        "estimator_name": name,
        "task": mlp_public_estimator_task(name),
        "solver": solver,
        "optimizer_boundary": mlp_public_optimizer_boundary(name, solver),
        "activation": str(getattr(estimator, "activation")),
        "out_activation": str(getattr(estimator, "out_activation_")),
        "hidden_layer_sizes": tuple(np.atleast_1d(getattr(estimator, "hidden_layer_sizes")).astype(int).tolist()),
        "layer_count": int(getattr(estimator, "n_layers_")),
        "output_count": int(getattr(estimator, "n_outputs_")),
        "n_iter": int(getattr(estimator, "n_iter_")),
        "coef_shapes": tuple(tuple(int(dim) for dim in coef.shape) for coef in getattr(estimator, "coefs_")),
        "intercept_shapes": tuple(tuple(int(dim) for dim in intercept.shape) for intercept in getattr(estimator, "intercepts_")),
    }
    if hasattr(estimator, "n_features_in_"):
        state["n_features_in"] = int(getattr(estimator, "n_features_in_"))
    if hasattr(estimator, "t_"):
        state["sample_counter"] = int(getattr(estimator, "t_"))
    if hasattr(estimator, "loss_"):
        state["loss"] = float(getattr(estimator, "loss_"))
    if hasattr(estimator, "best_loss_"):
        best_loss = getattr(estimator, "best_loss_")
        state["best_loss"] = None if best_loss is None else float(best_loss)
    if hasattr(estimator, "loss_curve_"):
        state["loss_curve_length"] = len(getattr(estimator, "loss_curve_"))
    if hasattr(estimator, "validation_scores_"):
        state["validation_score_count"] = len(getattr(estimator, "validation_scores_"))
    if hasattr(estimator, "best_validation_score_"):
        best_score = getattr(estimator, "best_validation_score_")
        state["best_validation_score"] = None if best_score is None else float(best_score)
    if hasattr(estimator, "classes_"):
        state["class_count"] = int(np.asarray(getattr(estimator, "classes_")).shape[0])
    return state
