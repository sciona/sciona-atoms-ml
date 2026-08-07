"""Ghost witnesses for public MLP estimator API-shell atoms."""

from __future__ import annotations


def witness_mlp_public_estimator_catalog(catalog_scope: str = "public_estimators") -> tuple[str, ...]:
    """Describe public MLP estimator names covered by this shell."""
    del catalog_scope
    return ("MLPClassifier", "MLPRegressor")


def witness_mlp_public_estimator_task(estimator_name: str) -> str:
    """Describe the public MLP estimator task family."""
    return "classification" if estimator_name == "MLPClassifier" else "regression"


def witness_mlp_public_optimizer_boundary(estimator_name: str, solver: str) -> str:
    """Describe the optimizer boundary behind a public MLP estimator."""
    del estimator_name
    if solver == "lbfgs":
        return "scipy_lbfgs_optimizer"
    if solver == "sgd":
        return "stochastic_sgd_optimizer"
    return "stochastic_adam_optimizer"


def witness_mlp_public_estimator_methods(estimator_name: str) -> tuple[str, ...]:
    """Describe public methods exposed for MLP routing."""
    return (estimator_name,)


def witness_mlp_public_fit_method_payload(
    estimator: object,
    X: object,
    y: object,
    *,
    sample_weight: object = None,
) -> dict[str, object]:
    """Describe a public MLP fit callback payload."""
    kwargs: dict[str, object] = {}
    if sample_weight is not None:
        kwargs["sample_weight"] = sample_weight
    return {"estimator": estimator, "method_name": "fit", "args": (X, y), "kwargs": kwargs}


def witness_mlp_public_prediction_method_payload(estimator: object, method_name: str, X: object) -> dict[str, object]:
    """Describe a public MLP prediction-like callback payload."""
    return {"estimator": estimator, "method_name": method_name, "args": (X,), "kwargs": {}}


def witness_mlp_public_fit_return_self(estimator: object) -> object:
    """Describe public MLP fit returning the fitted estimator."""
    return estimator


def witness_mlp_public_fitted_state_summary(estimator: object) -> dict[str, object]:
    """Describe fitted MLP state after optimizer callbacks."""
    return {"estimator": estimator}
