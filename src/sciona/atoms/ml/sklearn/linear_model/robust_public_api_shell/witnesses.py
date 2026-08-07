"""Ghost witnesses for public robust linear-model API-shell atoms."""

from __future__ import annotations


def witness_robust_linear_estimator_catalog(catalog_scope: str = "public_estimators") -> tuple[str, ...]:
    """Describe the public robust linear-model estimator names covered by this shell."""
    del catalog_scope
    return ("HuberRegressor", "QuantileRegressor", "RANSACRegressor")


def witness_robust_linear_estimator_family(estimator_name: str) -> str:
    """Describe the robust linear-model estimator family."""
    if estimator_name == "HuberRegressor":
        return "huber"
    if estimator_name == "QuantileRegressor":
        return "quantile"
    return "ransac"


def witness_robust_linear_estimator_boundary(estimator_name: str) -> str:
    """Describe the optimizer or callback boundary behind a robust estimator."""
    if estimator_name == "HuberRegressor":
        return "scipy_lbfgs"
    if estimator_name == "QuantileRegressor":
        return "scipy_linprog"
    return "estimator_consensus_callbacks"


def witness_robust_linear_estimator_methods(estimator_name: str) -> tuple[str, ...]:
    """Describe public methods exposed for framework-level robust estimator routing."""
    return (estimator_name,)


def witness_robust_linear_fit_method_payload(
    estimator: object,
    X: object,
    y: object,
    *,
    sample_weight: object = None,
    fit_params: dict[str, object] | None = None,
) -> dict[str, object]:
    """Describe a robust linear-model fit callback payload."""
    kwargs: dict[str, object] = {}
    if sample_weight is not None:
        kwargs["sample_weight"] = sample_weight
    if fit_params is not None:
        kwargs.update(fit_params)
    return {"estimator": estimator, "method_name": "fit", "args": (X, y), "kwargs": kwargs}


def witness_robust_linear_prediction_method_payload(estimator: object, method_name: str, X: object) -> dict[str, object]:
    """Describe a robust linear-model prediction-like callback payload."""
    return {"estimator": estimator, "method_name": method_name, "args": (X,), "kwargs": {}}


def witness_robust_linear_fit_return_self(estimator: object) -> object:
    """Describe public robust estimator fit returning the fitted estimator object."""
    return estimator


def witness_robust_linear_fitted_state_summary(estimator: object) -> dict[str, object]:
    """Describe fitted robust-estimator state exposed after external boundaries."""
    return {"estimator": estimator}
