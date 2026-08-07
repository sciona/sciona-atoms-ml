"""Ghost witnesses for public logistic estimator API-shell atoms."""

from __future__ import annotations


def witness_logistic_public_estimator_catalog(catalog_scope: str = "public_estimators") -> tuple[str, ...]:
    """Describe public logistic estimator names covered by this shell."""
    del catalog_scope
    return ("LogisticRegression", "LogisticRegressionCV")


def witness_logistic_public_estimator_family(estimator_name: str) -> str:
    """Describe the public logistic estimator family."""
    if estimator_name == "LogisticRegressionCV":
        return "logistic_cv"
    return "logistic"


def witness_logistic_public_solver_boundary(estimator_name: str, solver: str) -> str:
    """Describe the optimizer or solver boundary selected by a logistic estimator."""
    del estimator_name
    if solver == "liblinear":
        return "liblinear_native"
    if solver in {"sag", "saga"}:
        return "sag_saga_native"
    return "scipy_or_sklearn_newton_optimizer"


def witness_logistic_public_estimator_methods(estimator_name: str) -> tuple[str, ...]:
    """Describe public methods exposed for framework-level logistic routing."""
    return (estimator_name,)


def witness_logistic_public_fit_method_payload(
    estimator: object,
    X: object,
    y: object,
    *,
    sample_weight: object = None,
    params: dict[str, object] | None = None,
) -> dict[str, object]:
    """Describe a public logistic fit callback payload."""
    kwargs: dict[str, object] = {}
    if sample_weight is not None:
        kwargs["sample_weight"] = sample_weight
    if params is not None:
        kwargs.update(params)
    return {"estimator": estimator, "method_name": "fit", "args": (X, y), "kwargs": kwargs}


def witness_logistic_public_prediction_method_payload(estimator: object, method_name: str, X: object) -> dict[str, object]:
    """Describe a logistic prediction-like callback payload."""
    return {"estimator": estimator, "method_name": method_name, "args": (X,), "kwargs": {}}


def witness_logistic_public_fit_return_self(estimator: object) -> object:
    """Describe public logistic fit returning the fitted estimator object."""
    return estimator


def witness_logistic_public_fitted_state_summary(estimator: object) -> dict[str, object]:
    """Describe fitted logistic state exposed after solver or CV boundaries."""
    return {"estimator": estimator}
