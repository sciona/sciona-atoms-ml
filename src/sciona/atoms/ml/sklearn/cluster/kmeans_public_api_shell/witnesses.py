"""Ghost witnesses for public sklearn KMeans-family API-shell atoms."""

from __future__ import annotations


def witness_kmeans_estimator_catalog(
    catalog_scope: str = "public_estimators",
) -> tuple[str, ...]:
    """Describe the public KMeans-family estimator names covered by this shell."""
    del catalog_scope
    return ("KMeans", "MiniBatchKMeans", "BisectingKMeans")


def witness_kmeans_estimator_backend(estimator_name: str) -> str:
    """Describe the compiled solver family behind a public KMeans estimator."""
    if estimator_name == "MiniBatchKMeans":
        return "minibatch_kmeans_native"
    if estimator_name == "BisectingKMeans":
        return "recursive_kmeans_native"
    return "lloyd_elkan_native"


def witness_kmeans_estimator_task(estimator_name: str) -> str:
    """Describe the learning task exposed by a public KMeans estimator."""
    del estimator_name
    return "clustering"


def witness_kmeans_estimator_methods(estimator_name: str) -> tuple[str, ...]:
    """Describe public methods exposed for framework-level KMeans routing."""
    return (estimator_name,)


def witness_kmeans_fit_method_payload(
    estimator: object,
    X: object,
    *,
    sample_weight: object = None,
) -> dict[str, object]:
    """Describe a public KMeans fit-method callback payload."""
    kwargs = {} if sample_weight is None else {"sample_weight": sample_weight}
    return {"estimator": estimator, "method_name": "fit", "args": (X,), "kwargs": kwargs}


def witness_kmeans_prediction_method_payload(estimator: object, method_name: str, X: object) -> dict[str, object]:
    """Describe a public KMeans prediction-like method callback payload."""
    return {"estimator": estimator, "method_name": method_name, "args": (X,), "kwargs": {}}


def witness_kmeans_fit_return_self(estimator: object) -> object:
    """Describe public KMeans fit returning the fitted estimator object."""
    return estimator


def witness_kmeans_fitted_state(estimator: object) -> dict[str, object]:
    """Describe fitted center and label state exposed after KMeans fitting."""
    return {"estimator": estimator}

