"""Ghost witnesses for public multioutput meta-estimator API-shell atoms."""

from __future__ import annotations


def witness_multioutput_estimator_catalog(catalog_scope: str = "public_estimators") -> tuple[str, ...]:
    """Describe public multioutput meta-estimator names covered by this shell."""
    del catalog_scope
    return ("MultiOutputClassifier", "MultiOutputRegressor", "ClassifierChain", "RegressorChain")


def witness_multioutput_estimator_family(estimator_name: str) -> str:
    """Describe the public multioutput estimator family."""
    if estimator_name == "MultiOutputClassifier":
        return "multioutput_classifier"
    if estimator_name == "MultiOutputRegressor":
        return "multioutput_regressor"
    if estimator_name == "ClassifierChain":
        return "classifier_chain"
    return "regressor_chain"


def witness_multioutput_estimator_boundary(estimator_name: str) -> str:
    """Describe the callback boundary behind a public multioutput estimator."""
    if estimator_name in {"ClassifierChain", "RegressorChain"}:
        return "chain_estimator_cv_prediction_callbacks"
    return "per_output_estimator_callbacks"


def witness_multioutput_estimator_methods(estimator_name: str) -> tuple[str, ...]:
    """Describe public methods exposed for multioutput routing."""
    return (estimator_name,)


def witness_multioutput_fit_method_payload(
    estimator: object,
    X: object,
    y: object,
    *,
    sample_weight: object = None,
    fit_params: dict[str, object] | None = None,
) -> dict[str, object]:
    """Describe a public multioutput fit callback payload."""
    kwargs: dict[str, object] = {}
    if sample_weight is not None:
        kwargs["sample_weight"] = sample_weight
    if fit_params is not None:
        kwargs.update(fit_params)
    return {"estimator": estimator, "method_name": "fit", "args": (X, y), "kwargs": kwargs}


def witness_multioutput_prediction_method_payload(estimator: object, method_name: str, X: object) -> dict[str, object]:
    """Describe a public multioutput prediction-like callback payload."""
    return {"estimator": estimator, "method_name": method_name, "args": (X,), "kwargs": {}}


def witness_multioutput_fit_return_self(estimator: object) -> object:
    """Describe public multioutput fit returning the fitted estimator."""
    return estimator


def witness_multioutput_fitted_state_summary(estimator: object) -> dict[str, object]:
    """Describe fitted multioutput state after delegated callbacks."""
    return {"estimator": estimator}
