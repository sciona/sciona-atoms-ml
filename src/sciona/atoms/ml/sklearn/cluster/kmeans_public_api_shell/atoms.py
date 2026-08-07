"""Public sklearn KMeans-family API-shell atoms adapted from scikit-learn."""

from __future__ import annotations

from numbers import Integral

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_kmeans_estimator_backend,
    witness_kmeans_estimator_catalog,
    witness_kmeans_estimator_methods,
    witness_kmeans_estimator_task,
    witness_kmeans_fit_method_payload,
    witness_kmeans_fit_return_self,
    witness_kmeans_fitted_state,
    witness_kmeans_prediction_method_payload,
)

_ESTIMATOR_NAMES = ("KMeans", "MiniBatchKMeans", "BisectingKMeans")
_BASE_METHODS = {
    "KMeans": ("fit", "fit_predict", "fit_transform", "predict", "transform", "score"),
    "MiniBatchKMeans": ("fit", "partial_fit", "fit_predict", "fit_transform", "predict", "transform", "score"),
    "BisectingKMeans": ("fit", "fit_predict", "predict", "transform", "score"),
}


def _known_estimator_name(value: object) -> bool:
    return value in _ESTIMATOR_NAMES


def _positive_integral(value: object) -> bool:
    return bool(isinstance(value, Integral) and not isinstance(value, bool) and int(value) >= 1)


def _finite_dense_matrix(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _sample_weight_valid(sample_weight: object, X: object) -> bool:
    if sample_weight is None:
        return True
    try:
        weights = np.asarray(sample_weight, dtype=np.float64)
        matrix = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        matrix.ndim == 2
        and weights.ndim == 1
        and weights.shape == (matrix.shape[0],)
        and np.all(np.isfinite(weights))
        and np.all(weights >= 0.0)
        and np.any(weights > 0.0)
    )


def _public_kmeans_estimator(estimator: object) -> bool:
    from sklearn.cluster import BisectingKMeans, KMeans, MiniBatchKMeans

    return isinstance(estimator, (KMeans, MiniBatchKMeans, BisectingKMeans))


def _fitted_public_kmeans(estimator: object) -> bool:
    return bool(
        _public_kmeans_estimator(estimator)
        and hasattr(estimator, "cluster_centers_")
        and hasattr(estimator, "labels_")
        and hasattr(estimator, "inertia_")
        and hasattr(estimator, "n_features_in_")
    )


def _fit_input_valid(estimator: object, X: object) -> bool:
    if not _public_kmeans_estimator(estimator) or not _finite_dense_matrix(X):
        return False
    n_clusters = getattr(estimator, "n_clusters", None)
    if not _positive_integral(n_clusters):
        return False
    return bool(np.asarray(X, dtype=np.float64).shape[0] >= int(n_clusters))


def _method_available(estimator: object, method_name: str) -> bool:
    return bool(isinstance(method_name, str) and method_name != "" and hasattr(estimator, method_name))


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
    )


def _finite_array(value: object) -> bool:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return bool(array.size >= 1 and np.all(np.isfinite(array)))


@register_atom(witness_kmeans_estimator_catalog)
@icontract.require(lambda catalog_scope: catalog_scope == "public_estimators", "catalog_scope must be 'public_estimators'")
@icontract.ensure(lambda result: result == _ESTIMATOR_NAMES, "catalog must expose the public KMeans-family estimators")
def kmeans_estimator_catalog(
    catalog_scope: str = "public_estimators",
) -> tuple[str, ...]:
    """Expose public sklearn.cluster KMeans-family estimator names for selection."""
    del catalog_scope
    return _ESTIMATOR_NAMES


@register_atom(witness_kmeans_estimator_backend)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered KMeans estimator")
@icontract.ensure(
    lambda result: result in {"lloyd_elkan_native", "minibatch_kmeans_native", "recursive_kmeans_native"},
    "backend boundary must name the KMeans solver family",
)
def kmeans_estimator_backend(estimator_name: str) -> str:
    """Return the native solver boundary family behind a public KMeans estimator."""
    if estimator_name == "MiniBatchKMeans":
        return "minibatch_kmeans_native"
    if estimator_name == "BisectingKMeans":
        return "recursive_kmeans_native"
    return "lloyd_elkan_native"


@register_atom(witness_kmeans_estimator_task)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered KMeans estimator")
@icontract.ensure(lambda result: result == "clustering", "KMeans-family estimators expose clustering")
def kmeans_estimator_task(estimator_name: str) -> str:
    """Return the high-level learning task for a public KMeans-family estimator."""
    del estimator_name
    return "clustering"


@register_atom(witness_kmeans_estimator_methods)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered KMeans estimator")
@icontract.ensure(lambda result: isinstance(result, tuple) and "fit" in result and "predict" in result, "methods must include fit and predict")
def kmeans_estimator_methods(estimator_name: str) -> tuple[str, ...]:
    """Expose public methods useful for high-level KMeans-family routing."""
    return _BASE_METHODS[estimator_name]


@register_atom(witness_kmeans_fit_method_payload)
@icontract.require(lambda estimator, X: _fit_input_valid(estimator, X), "estimator and finite dense X must satisfy the KMeans fit boundary")
@icontract.require(lambda sample_weight, X: _sample_weight_valid(sample_weight, X), "sample_weight must match the sample axis when provided")
@icontract.ensure(lambda result: isinstance(result, dict) and result["method_name"] == "fit", "payload must target public fit")
@icontract.ensure(
    lambda result, estimator, X: _payload_result_valid(result, estimator, "fit", X),
    "fit payload must preserve estimator and positional input",
)
def kmeans_fit_method_payload(
    estimator: object,
    X: object,
    *,
    sample_weight: object = None,
) -> dict[str, object]:
    """Expose a public KMeans fit payload without executing the native solver."""
    kwargs = {} if sample_weight is None else {"sample_weight": sample_weight}
    return {"estimator": estimator, "method_name": "fit", "args": (X,), "kwargs": kwargs}


@register_atom(witness_kmeans_prediction_method_payload)
@icontract.require(lambda estimator: _fitted_public_kmeans(estimator), "estimator must be a fitted public KMeans-family estimator")
@icontract.require(lambda X: _finite_dense_matrix(X), "X must be a finite dense 2D array at the prediction boundary")
@icontract.require(lambda estimator, method_name, X: _method_available(estimator, method_name), "method_name must be available on the fitted estimator")
@icontract.ensure(
    lambda result, estimator, method_name, X: _payload_result_valid(result, estimator, method_name, X),
    "prediction payload must preserve estimator, method, positional input, and empty kwargs",
)
def kmeans_prediction_method_payload(estimator: object, method_name: str, X: object) -> dict[str, object]:
    """Expose a public KMeans prediction-like method payload without executing it."""
    return {"estimator": estimator, "method_name": method_name, "args": (X,), "kwargs": {}}


@register_atom(witness_kmeans_fit_return_self)
@icontract.require(lambda estimator: _fitted_public_kmeans(estimator), "estimator must be a fitted public KMeans-family estimator")
@icontract.ensure(lambda result, estimator: result is estimator and _fitted_public_kmeans(result), "fit shell must return fitted self")
def kmeans_fit_return_self(estimator: object) -> object:
    """Return the fitted KMeans-family estimator from the public fit shell."""
    return estimator


@register_atom(witness_kmeans_fitted_state)
@icontract.require(lambda estimator: _fitted_public_kmeans(estimator), "estimator must be a fitted public KMeans-family estimator")
@icontract.ensure(lambda result: isinstance(result, dict) and result["backend"].endswith("_native"), "state payload must identify a native solver boundary")
@icontract.ensure(lambda result: _finite_array(result["cluster_centers"]) and _finite_array(result["inertia"]), "state must expose finite centers and inertia")
def kmeans_fitted_state(estimator: object) -> dict[str, object]:
    """Expose fitted KMeans centers, labels, and inertia after the native solver."""
    state: dict[str, object] = {
        "backend": kmeans_estimator_backend(estimator.__class__.__name__),
        "estimator_name": estimator.__class__.__name__,
        "cluster_centers": np.asarray(getattr(estimator, "cluster_centers_")),
        "labels": np.asarray(getattr(estimator, "labels_")),
        "inertia": float(getattr(estimator, "inertia_")),
        "n_features_in": int(getattr(estimator, "n_features_in_")),
        "n_clusters": int(getattr(estimator, "n_clusters")),
    }
    if hasattr(estimator, "n_iter_"):
        state["n_iter"] = int(getattr(estimator, "n_iter_"))
    if hasattr(estimator, "_n_threads"):
        state["n_threads"] = int(getattr(estimator, "_n_threads"))
    return state

