"""Public sklearn SVM API-shell atoms adapted from scikit-learn."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_svm_estimator_backend,
    witness_svm_estimator_catalog,
    witness_svm_estimator_methods,
    witness_svm_estimator_task,
    witness_svm_fit_return_self,
    witness_svm_liblinear_fitted_state,
    witness_svm_libsvm_fitted_support_state,
    witness_svm_linear_fit_liblinear_payload,
    witness_svm_prediction_method_payload,
)

_ESTIMATOR_NAMES = ("SVC", "NuSVC", "SVR", "NuSVR", "LinearSVC", "LinearSVR", "OneClassSVM")
_LIBLINEAR_ESTIMATORS = {"LinearSVC", "LinearSVR"}
_LIBSVM_ESTIMATORS = {"SVC", "NuSVC", "SVR", "NuSVR", "OneClassSVM"}
_CLASSIFIERS = {"SVC", "NuSVC", "LinearSVC"}
_REGRESSORS = {"SVR", "NuSVR", "LinearSVR"}
_OUTLIER_DETECTORS = {"OneClassSVM"}
_LINEAR_LOSSES = {"hinge", "squared_hinge", "epsilon_insensitive", "squared_epsilon_insensitive"}
_LINEAR_PENALTIES = {"l1", "l2"}
_MULTI_CLASS_VALUES = {"ovr", "crammer_singer"}
_BASE_METHODS = {
    "SVC": ("fit", "predict", "decision_function", "score"),
    "NuSVC": ("fit", "predict", "decision_function", "score"),
    "LinearSVC": ("fit", "predict", "decision_function", "score"),
    "SVR": ("fit", "predict", "score"),
    "NuSVR": ("fit", "predict", "score"),
    "LinearSVR": ("fit", "predict", "score"),
    "OneClassSVM": ("fit", "fit_predict", "predict", "decision_function", "score_samples"),
}


def _known_estimator_name(value: object) -> bool:
    return value in _ESTIMATOR_NAMES


def _libsvm_estimator_name(value: object) -> bool:
    return value in _LIBSVM_ESTIMATORS


def _linear_estimator_name(value: object) -> bool:
    return value in _LIBLINEAR_ESTIMATORS


def _bool_scalar(value: object) -> bool:
    return isinstance(value, (bool, np.bool_))


def _positive_real(value: object) -> bool:
    return bool(isinstance(value, Real) and not isinstance(value, bool) and float(value) > 0.0)


def _nonnegative_real(value: object) -> bool:
    return bool(isinstance(value, Real) and not isinstance(value, bool) and float(value) >= 0.0)


def _integral(value: object) -> bool:
    return bool(isinstance(value, Integral) and not isinstance(value, bool))


def _nonnegative_integral(value: object) -> bool:
    return bool(_integral(value) and int(value) >= 0)


def _finite_array(value: object) -> bool:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return bool(array.size >= 1 and np.all(np.isfinite(array)))


def _fitted_svm_estimator(estimator: object) -> bool:
    from sklearn.svm import LinearSVC, LinearSVR, NuSVC, NuSVR, OneClassSVM, SVC, SVR

    if isinstance(estimator, (LinearSVC, LinearSVR)):
        return bool(hasattr(estimator, "coef_") and hasattr(estimator, "intercept_") and hasattr(estimator, "n_iter_"))
    if isinstance(estimator, (SVC, NuSVC, SVR, NuSVR, OneClassSVM)):
        return bool(hasattr(estimator, "support_") and hasattr(estimator, "support_vectors_") and hasattr(estimator, "intercept_"))
    return False


def _fitted_libsvm_estimator(estimator: object) -> bool:
    from sklearn.svm import NuSVC, NuSVR, OneClassSVM, SVC, SVR

    return bool(isinstance(estimator, (SVC, NuSVC, SVR, NuSVR, OneClassSVM)) and _fitted_svm_estimator(estimator))


def _fitted_liblinear_estimator(estimator: object) -> bool:
    from sklearn.svm import LinearSVC, LinearSVR

    return bool(isinstance(estimator, (LinearSVC, LinearSVR)) and _fitted_svm_estimator(estimator))


def _method_available(estimator: object, method_name: str, check_probability: bool) -> bool:
    if not isinstance(method_name, str) or method_name == "":
        return False
    if not hasattr(estimator, method_name):
        return False
    if check_probability and method_name in {"predict_proba", "predict_log_proba"}:
        return bool(getattr(estimator, "probability", False))
    return True


def _payload_result_valid(result: object, estimator: object, method_name: str, X: object) -> bool:
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


@register_atom(witness_svm_estimator_catalog)
@icontract.require(lambda catalog_scope: catalog_scope == "public_estimators", "catalog_scope must be 'public_estimators'")
@icontract.ensure(lambda result: result == _ESTIMATOR_NAMES, "catalog must expose the seven public SVM estimators")
def svm_estimator_catalog(
    catalog_scope: str = "public_estimators",
) -> tuple[str, ...]:
    """Expose public sklearn.svm estimator names for framework selection."""
    del catalog_scope
    return _ESTIMATOR_NAMES


@register_atom(witness_svm_estimator_backend)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered SVM estimator")
@icontract.ensure(lambda result: result in {"libsvm", "liblinear"}, "backend must be libsvm or liblinear")
def svm_estimator_backend(estimator_name: str) -> str:
    """Return the compiled backend family used by a public SVM estimator."""
    if estimator_name in _LIBLINEAR_ESTIMATORS:
        return "liblinear"
    return "libsvm"


@register_atom(witness_svm_estimator_task)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered SVM estimator")
@icontract.ensure(
    lambda result: result in {"classification", "regression", "outlier_detection"},
    "task must be one of the covered high-level SVM tasks",
)
def svm_estimator_task(estimator_name: str) -> str:
    """Return the high-level learning task for a public SVM estimator."""
    if estimator_name in _CLASSIFIERS:
        return "classification"
    if estimator_name in _REGRESSORS:
        return "regression"
    return "outlier_detection"


@register_atom(witness_svm_estimator_methods)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered SVM estimator")
@icontract.require(lambda probability_enabled: _bool_scalar(probability_enabled), "probability_enabled must be boolean")
@icontract.ensure(lambda result: isinstance(result, tuple) and "fit" in result and "predict" in result, "methods must include fit and predict")
def svm_estimator_methods(
    estimator_name: str,
    *,
    probability_enabled: bool = False,
) -> tuple[str, ...]:
    """Expose public methods useful for high-level SVM routing."""
    methods = _BASE_METHODS[estimator_name]
    if estimator_name in {"SVC", "NuSVC"} and probability_enabled:
        return methods + ("predict_proba", "predict_log_proba")
    return methods


@register_atom(witness_svm_linear_fit_liblinear_payload)
@icontract.require(lambda estimator_name: _linear_estimator_name(estimator_name), "estimator_name must be LinearSVC or LinearSVR")
@icontract.require(lambda C: _positive_real(C), "C must be positive")
@icontract.require(lambda fit_intercept: _bool_scalar(fit_intercept), "fit_intercept must be boolean")
@icontract.require(lambda intercept_scaling: _positive_real(intercept_scaling), "intercept_scaling must be positive")
@icontract.require(lambda penalty: penalty in _LINEAR_PENALTIES, "penalty must be a supported liblinear penalty")
@icontract.require(lambda dual: _bool_scalar(dual), "dual must be boolean after auto resolution")
@icontract.require(lambda verbose: _nonnegative_integral(verbose), "verbose must be a nonnegative integer")
@icontract.require(lambda max_iter: _nonnegative_integral(max_iter), "max_iter must be a nonnegative integer")
@icontract.require(lambda tol: _positive_real(tol), "tol must be positive")
@icontract.require(lambda multi_class: multi_class in _MULTI_CLASS_VALUES, "multi_class must be a known liblinear mode")
@icontract.require(lambda loss: loss in _LINEAR_LOSSES, "loss must be a known linear SVM loss")
@icontract.require(lambda epsilon: _nonnegative_real(epsilon), "epsilon must be nonnegative")
@icontract.ensure(lambda result: isinstance(result, dict) and result["backend"] == "liblinear", "payload must target liblinear")
def svm_linear_fit_liblinear_payload(
    estimator_name: str,
    C: float,
    fit_intercept: bool,
    intercept_scaling: float,
    class_weight: object,
    penalty: str,
    dual: bool,
    verbose: int,
    max_iter: int,
    tol: float,
    random_state: object,
    multi_class: str,
    loss: str,
    *,
    epsilon: float = 0.0,
    sample_weight: object = None,
) -> dict[str, object]:
    """Expose the _fit_liblinear payload assembled by LinearSVC/LinearSVR.fit."""
    payload: dict[str, object] = {
        "backend": "liblinear",
        "estimator_name": estimator_name,
        "C": float(C),
        "fit_intercept": bool(fit_intercept),
        "intercept_scaling": float(intercept_scaling),
        "class_weight": class_weight,
        "penalty": penalty,
        "dual": bool(dual),
        "verbose": int(verbose),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "random_state": random_state,
        "multi_class": multi_class,
        "loss": loss,
        "epsilon": float(epsilon),
        "sample_weight": sample_weight,
    }
    if estimator_name == "LinearSVR":
        payload["class_weight"] = None
        payload["penalty"] = "l2"
        payload["multi_class"] = "ovr"
    return payload


@register_atom(witness_svm_prediction_method_payload)
@icontract.require(lambda estimator: _fitted_svm_estimator(estimator), "estimator must be a fitted public SVM estimator")
@icontract.require(
    lambda estimator, method_name, X, check_probability: _method_available(estimator, method_name, bool(check_probability)),
    "method_name must be available on the fitted estimator",
)
@icontract.ensure(
    lambda result, estimator, method_name, X: _payload_result_valid(result, estimator, method_name, X),
    "prediction payload must preserve estimator, method, positional input, and empty kwargs",
)
def svm_prediction_method_payload(
    estimator: object,
    method_name: str,
    X: object,
    *,
    check_probability: bool = True,
) -> dict[str, object]:
    """Expose a public SVM prediction-like method payload without executing it."""
    del check_probability
    return {"estimator": estimator, "method_name": method_name, "args": (X,), "kwargs": {}}


@register_atom(witness_svm_fit_return_self)
@icontract.require(lambda estimator: _fitted_svm_estimator(estimator), "estimator must be a fitted public SVM estimator")
@icontract.ensure(lambda result, estimator: result is estimator and _fitted_svm_estimator(result), "fit shell must return fitted self")
def svm_fit_return_self(estimator: object) -> object:
    """Return the fitted SVM estimator from the public fit shell."""
    return estimator


@register_atom(witness_svm_libsvm_fitted_support_state)
@icontract.require(lambda estimator: _fitted_libsvm_estimator(estimator), "estimator must be a fitted libsvm-backed SVM estimator")
@icontract.ensure(lambda result: isinstance(result, dict) and result["backend"] == "libsvm", "state payload must identify libsvm")
def svm_libsvm_fitted_support_state(estimator: object) -> dict[str, Any]:
    """Expose fitted support-vector state after the deferred libsvm solver."""
    state: dict[str, Any] = {
        "backend": "libsvm",
        "estimator_name": estimator.__class__.__name__,
        "support": np.asarray(getattr(estimator, "support_")),
        "support_vectors": np.asarray(getattr(estimator, "support_vectors_")),
        "dual_coef": np.asarray(getattr(estimator, "dual_coef_")),
        "intercept": np.asarray(getattr(estimator, "intercept_")),
        "n_support": np.asarray(getattr(estimator, "n_support_")),
    }
    if hasattr(estimator, "classes_"):
        state["classes"] = np.asarray(getattr(estimator, "classes_"))
    if hasattr(estimator, "fit_status_"):
        state["fit_status"] = int(getattr(estimator, "fit_status_"))
    if hasattr(estimator, "offset_"):
        state["offset"] = float(np.asarray(getattr(estimator, "offset_")).reshape(-1)[0])
    return state


@register_atom(witness_svm_liblinear_fitted_state)
@icontract.require(lambda estimator: _fitted_liblinear_estimator(estimator), "estimator must be a fitted liblinear-backed SVM estimator")
@icontract.ensure(lambda result: isinstance(result, dict) and result["backend"] == "liblinear", "state payload must identify liblinear")
@icontract.ensure(lambda result: _finite_array(result["coef"]) and _finite_array(result["intercept"]), "linear state must expose finite coefficients")
def svm_liblinear_fitted_state(estimator: object) -> dict[str, Any]:
    """Expose fitted coefficient state after the deferred liblinear solver."""
    state: dict[str, Any] = {
        "backend": "liblinear",
        "estimator_name": estimator.__class__.__name__,
        "coef": np.asarray(getattr(estimator, "coef_")),
        "intercept": np.asarray(getattr(estimator, "intercept_")),
        "n_iter": int(getattr(estimator, "n_iter_")),
    }
    if hasattr(estimator, "classes_"):
        state["classes"] = np.asarray(getattr(estimator, "classes_"))
    return state
