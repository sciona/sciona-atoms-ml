"""Ghost witnesses for public multiclass meta-estimator API-shell atoms."""

from __future__ import annotations


def witness_multiclass_estimator_catalog(catalog_scope: str = "public_estimators") -> tuple[str, ...]:
    """Describe public multiclass meta-estimator names covered by this shell."""
    del catalog_scope
    return ("OneVsRestClassifier", "OneVsOneClassifier", "OutputCodeClassifier")


def witness_multiclass_estimator_family(estimator_name: str) -> str:
    """Describe the public multiclass estimator family."""
    if estimator_name == "OneVsRestClassifier":
        return "one_vs_rest"
    if estimator_name == "OneVsOneClassifier":
        return "one_vs_one"
    return "output_code"


def witness_multiclass_estimator_boundary(estimator_name: str) -> str:
    """Describe the callback boundary behind a public multiclass estimator."""
    if estimator_name == "OutputCodeClassifier":
        return "code_book_estimator_response_callbacks"
    return "cloned_estimator_response_callbacks"


def witness_multiclass_estimator_methods(estimator_name: str) -> tuple[str, ...]:
    """Describe public methods exposed for multiclass routing."""
    return (estimator_name,)


def witness_multiclass_fit_method_payload(
    estimator: object,
    X: object,
    y: object,
    *,
    fit_params: dict[str, object] | None = None,
) -> dict[str, object]:
    """Describe a public multiclass fit callback payload."""
    kwargs = {} if fit_params is None else dict(fit_params)
    return {"estimator": estimator, "method_name": "fit", "args": (X, y), "kwargs": kwargs}


def witness_multiclass_prediction_method_payload(estimator: object, method_name: str, X: object) -> dict[str, object]:
    """Describe a public multiclass prediction-like callback payload."""
    return {"estimator": estimator, "method_name": method_name, "args": (X,), "kwargs": {}}


def witness_multiclass_fit_return_self(estimator: object) -> object:
    """Describe public multiclass fit returning the fitted estimator."""
    return estimator


def witness_multiclass_fitted_state_summary(estimator: object) -> dict[str, object]:
    """Describe fitted multiclass state after delegated callbacks."""
    return {"estimator": estimator}
