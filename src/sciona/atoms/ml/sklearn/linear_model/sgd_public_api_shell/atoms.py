"""Public SGD-family sklearn API-shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_sgd_public_estimator_catalog,
    witness_sgd_public_estimator_family,
    witness_sgd_public_estimator_methods,
    witness_sgd_public_estimator_task,
    witness_sgd_public_fit_method_payload,
    witness_sgd_public_fit_return_self,
    witness_sgd_public_fitted_state_summary,
    witness_sgd_public_prediction_method_payload,
    witness_sgd_public_training_boundary,
)

_ESTIMATOR_NAMES = (
    "SGDClassifier",
    "SGDRegressor",
    "SGDOneClassSVM",
    "PassiveAggressiveClassifier",
    "PassiveAggressiveRegressor",
    "Perceptron",
)
_FAMILIES = {
    "SGDClassifier": "sgd_classifier",
    "SGDRegressor": "sgd_regressor",
    "SGDOneClassSVM": "sgd_one_class_svm",
    "PassiveAggressiveClassifier": "passive_aggressive_classifier",
    "PassiveAggressiveRegressor": "passive_aggressive_regressor",
    "Perceptron": "perceptron",
}
_CLASSIFIERS = {"SGDClassifier", "PassiveAggressiveClassifier", "Perceptron"}
_REGRESSORS = {"SGDRegressor", "PassiveAggressiveRegressor"}
_NO_SAMPLE_WEIGHT = {"PassiveAggressiveClassifier", "PassiveAggressiveRegressor"}
_METHODS = {
    "SGDClassifier": ("fit", "partial_fit", "predict", "decision_function", "score"),
    "SGDRegressor": ("fit", "partial_fit", "predict", "score"),
    "SGDOneClassSVM": ("fit", "partial_fit", "predict", "decision_function"),
    "PassiveAggressiveClassifier": ("fit", "partial_fit", "predict", "decision_function", "score"),
    "PassiveAggressiveRegressor": ("fit", "partial_fit", "predict", "score"),
    "Perceptron": ("fit", "partial_fit", "predict", "decision_function", "score"),
}
_BOUNDARIES = {"compiled_plain_sgd", "compiled_one_class_plain_sgd"}


def _known_estimator_name(value: object) -> bool:
    return value in _ESTIMATOR_NAMES


def _finite_dense_matrix(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _target_valid(estimator: object, X: object, y: object) -> bool:
    name = estimator.__class__.__name__
    try:
        matrix = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if name == "SGDOneClassSVM":
        if y is None:
            return True
        targets = np.asarray(y)
        return bool(targets.ndim == 1 and matrix.ndim == 2 and targets.shape == (matrix.shape[0],))
    if y is None:
        return False
    try:
        targets = np.asarray(y)
    except (TypeError, ValueError):
        return False
    if not (matrix.ndim == 2 and targets.ndim == 1 and targets.shape == (matrix.shape[0],)):
        return False
    if name in _CLASSIFIERS:
        return bool(np.unique(targets).shape[0] >= 2)
    try:
        numeric_targets = np.asarray(y, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(np.all(np.isfinite(numeric_targets)))


def _sample_weight_valid(estimator: object, sample_weight: object, X: object) -> bool:
    if sample_weight is None:
        return True
    if estimator.__class__.__name__ in _NO_SAMPLE_WEIGHT:
        return False
    try:
        weights = np.asarray(sample_weight, dtype=np.float64)
        matrix = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(weights.ndim == 1 and matrix.ndim == 2 and weights.shape == (matrix.shape[0],) and np.all(np.isfinite(weights)) and np.all(weights >= 0.0))


def _init_array_valid(value: object, X: object) -> bool:
    if value is None:
        return True
    try:
        array = np.asarray(value, dtype=np.float64)
        matrix = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.size >= 1 and matrix.ndim == 2 and np.all(np.isfinite(array)))


def _offset_valid(estimator: object, offset_init: object) -> bool:
    if offset_init is None:
        return True
    if estimator.__class__.__name__ != "SGDOneClassSVM":
        return False
    try:
        return bool(np.isfinite(float(offset_init)))
    except (TypeError, ValueError):
        return False


def _intercept_valid(estimator: object, intercept_init: object, X: object) -> bool:
    if intercept_init is None:
        return True
    if estimator.__class__.__name__ == "SGDOneClassSVM":
        return False
    return _init_array_valid(intercept_init, X)


def _public_sgd_estimator(estimator: object) -> bool:
    from sklearn.linear_model import (
        PassiveAggressiveClassifier,
        PassiveAggressiveRegressor,
        Perceptron,
        SGDClassifier,
        SGDOneClassSVM,
        SGDRegressor,
    )

    return isinstance(
        estimator,
        (
            SGDClassifier,
            SGDRegressor,
            SGDOneClassSVM,
            PassiveAggressiveClassifier,
            PassiveAggressiveRegressor,
            Perceptron,
        ),
    )


def _fitted_public_sgd(estimator: object) -> bool:
    if not (_public_sgd_estimator(estimator) and hasattr(estimator, "coef_") and hasattr(estimator, "n_features_in_")):
        return False
    if estimator.__class__.__name__ == "SGDOneClassSVM":
        return bool(hasattr(estimator, "offset_") and hasattr(estimator, "t_"))
    return bool(hasattr(estimator, "intercept_") and hasattr(estimator, "n_iter_"))


def _fit_input_valid(estimator: object, X: object, y: object) -> bool:
    return bool(_public_sgd_estimator(estimator) and _finite_dense_matrix(X) and _target_valid(estimator, X, y))


def _method_available(estimator: object, method_name: str) -> bool:
    return bool(isinstance(method_name, str) and method_name != "" and hasattr(estimator, method_name))


def _fit_payload_valid(result: object, estimator: object, X: object, y: object) -> bool:
    if not isinstance(result, dict):
        return False
    args = result.get("args")
    kwargs = result.get("kwargs")
    expected_len = 1 if y is None else 2
    return bool(
        result.get("estimator") is estimator
        and result.get("method_name") == "fit"
        and isinstance(args, tuple)
        and len(args) == expected_len
        and args[0] is X
        and (expected_len == 1 or args[1] is y)
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


@register_atom(witness_sgd_public_estimator_catalog)
@icontract.require(lambda catalog_scope: catalog_scope == "public_estimators", "catalog_scope must be 'public_estimators'")
@icontract.ensure(lambda result: result == _ESTIMATOR_NAMES, "catalog must expose covered public SGD-family estimators")
def sgd_public_estimator_catalog(catalog_scope: str = "public_estimators") -> tuple[str, ...]:
    """Expose public SGD-family estimator names for framework selection."""
    del catalog_scope
    return _ESTIMATOR_NAMES


@register_atom(witness_sgd_public_estimator_family)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered public SGD-family estimator")
@icontract.ensure(lambda result: result in set(_FAMILIES.values()), "family must name a covered SGD-family estimator")
def sgd_public_estimator_family(estimator_name: str) -> str:
    """Return the public SGD-family estimator family."""
    return _FAMILIES[estimator_name]


@register_atom(witness_sgd_public_estimator_task)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered public SGD-family estimator")
@icontract.ensure(lambda result: result in {"classification", "regression", "anomaly_detection"}, "task must be covered by the SGD-family shell")
def sgd_public_estimator_task(estimator_name: str) -> str:
    """Return the learning task for a public SGD-family estimator."""
    if estimator_name in _REGRESSORS:
        return "regression"
    if estimator_name == "SGDOneClassSVM":
        return "anomaly_detection"
    return "classification"


@register_atom(witness_sgd_public_training_boundary)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered public SGD-family estimator")
@icontract.ensure(lambda result: result in _BOUNDARIES, "boundary must name a covered compiled SGD training family")
def sgd_public_training_boundary(estimator_name: str) -> str:
    """Return the compiled training boundary for a public SGD-family estimator."""
    if estimator_name == "SGDOneClassSVM":
        return "compiled_one_class_plain_sgd"
    return "compiled_plain_sgd"


@register_atom(witness_sgd_public_estimator_methods)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered public SGD-family estimator")
@icontract.ensure(lambda result: isinstance(result, tuple) and "fit" in result and "predict" in result, "methods must include fit and predict")
def sgd_public_estimator_methods(estimator_name: str) -> tuple[str, ...]:
    """Expose public methods useful for high-level SGD-family routing."""
    return _METHODS[estimator_name]


@register_atom(witness_sgd_public_fit_method_payload)
@icontract.require(lambda estimator, X, y: _fit_input_valid(estimator, X, y), "estimator and finite dense fit inputs must satisfy the SGD-family fit boundary")
@icontract.require(lambda estimator, sample_weight, X: _sample_weight_valid(estimator, sample_weight, X), "sample_weight must match the sample axis and be supported when provided")
@icontract.require(lambda coef_init, X: _init_array_valid(coef_init, X), "coef_init must be finite when provided")
@icontract.require(lambda estimator, intercept_init, X: _intercept_valid(estimator, intercept_init, X), "intercept_init must be finite and supported when provided")
@icontract.require(lambda estimator, offset_init: _offset_valid(estimator, offset_init), "offset_init is only supported for SGDOneClassSVM")
@icontract.ensure(lambda result: isinstance(result, dict) and result["method_name"] == "fit", "payload must target public fit")
@icontract.ensure(lambda result, estimator, X, y: _fit_payload_valid(result, estimator, X, y), "fit payload must preserve estimator and positional fit inputs")
def sgd_public_fit_method_payload(
    estimator: object,
    X: object,
    y: object = None,
    *,
    sample_weight: object = None,
    coef_init: object = None,
    intercept_init: object = None,
    offset_init: object = None,
) -> dict[str, object]:
    """Package a public SGD-family fit call without running compiled training."""
    kwargs: dict[str, object] = {}
    if sample_weight is not None:
        kwargs["sample_weight"] = sample_weight
    if coef_init is not None:
        kwargs["coef_init"] = coef_init
    if intercept_init is not None:
        kwargs["intercept_init"] = intercept_init
    if offset_init is not None:
        kwargs["offset_init"] = offset_init
    args = (X,) if y is None else (X, y)
    return {"estimator": estimator, "method_name": "fit", "args": args, "kwargs": kwargs}


@register_atom(witness_sgd_public_prediction_method_payload)
@icontract.require(lambda estimator: _fitted_public_sgd(estimator), "estimator must be a fitted public SGD-family estimator")
@icontract.require(lambda X: _finite_dense_matrix(X), "X must be a finite dense 2D array at the prediction boundary")
@icontract.require(lambda estimator, method_name, X: _method_available(estimator, method_name), "method_name must be available on the fitted estimator")
@icontract.ensure(
    lambda result, estimator, method_name, X: _prediction_payload_valid(result, estimator, method_name, X),
    "prediction payload must preserve estimator, method, positional input, and empty kwargs",
)
def sgd_public_prediction_method_payload(estimator: object, method_name: str, X: object) -> dict[str, object]:
    """Expose a public SGD-family prediction-like method payload without executing it."""
    return {"estimator": estimator, "method_name": method_name, "args": (X,), "kwargs": {}}


@register_atom(witness_sgd_public_fit_return_self)
@icontract.require(lambda estimator: _fitted_public_sgd(estimator), "estimator must be a fitted public SGD-family estimator")
@icontract.ensure(lambda result, estimator: result is estimator and _fitted_public_sgd(result), "fit shell must return fitted self")
def sgd_public_fit_return_self(estimator: object) -> object:
    """Return the fitted SGD-family estimator from the public fit shell."""
    return estimator


@register_atom(witness_sgd_public_fitted_state_summary)
@icontract.require(lambda estimator: _fitted_public_sgd(estimator), "estimator must be a fitted public SGD-family estimator")
@icontract.ensure(lambda result: isinstance(result, dict) and result["training_boundary"] in _BOUNDARIES, "summary must expose training-boundary metadata")
@icontract.ensure(lambda result: _finite_array(result["coef"]), "summary must expose finite fitted coefficients")
def sgd_public_fitted_state_summary(estimator: object) -> dict[str, object]:
    """Expose compact fitted SGD-family estimator state after compiled training."""
    name = estimator.__class__.__name__
    state: dict[str, object] = {
        "estimator_name": name,
        "family": sgd_public_estimator_family(name),
        "task": sgd_public_estimator_task(name),
        "training_boundary": sgd_public_training_boundary(name),
        "coef": np.asarray(getattr(estimator, "coef_")),
        "n_features_in": int(getattr(estimator, "n_features_in_")),
    }
    if hasattr(estimator, "intercept_"):
        state["intercept"] = np.asarray(getattr(estimator, "intercept_"))
    if hasattr(estimator, "classes_"):
        state["classes"] = tuple(np.asarray(getattr(estimator, "classes_")).tolist())
        state["class_count"] = int(np.asarray(getattr(estimator, "classes_")).shape[0])
    if hasattr(estimator, "n_iter_"):
        state["n_iter"] = int(getattr(estimator, "n_iter_"))
    if hasattr(estimator, "t_"):
        state["t"] = float(getattr(estimator, "t_"))
    if hasattr(estimator, "offset_"):
        state["offset"] = np.asarray(getattr(estimator, "offset_"))
    if hasattr(estimator, "loss"):
        state["loss"] = getattr(estimator, "loss")
    if hasattr(estimator, "penalty"):
        state["penalty"] = getattr(estimator, "penalty")
    if name in {"PassiveAggressiveClassifier", "PassiveAggressiveRegressor"}:
        state["deprecated_since"] = "1.8"
        state["removed_in"] = "1.10"
    return state
