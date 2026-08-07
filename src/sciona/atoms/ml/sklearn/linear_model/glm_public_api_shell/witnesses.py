"""Ghost witnesses for public sklearn GLM regressor API-shell atoms."""

from __future__ import annotations


def witness_glm_estimator_catalog(catalog_scope: str = "public_estimators") -> tuple[str, ...]:
    """Describe the public sklearn GLM estimator names covered by this shell."""
    del catalog_scope
    return ("PoissonRegressor", "GammaRegressor", "TweedieRegressor")


def witness_glm_estimator_distribution(estimator_name: str, power: float | None = None) -> str:
    """Describe the response distribution represented by a public GLM estimator."""
    del power
    if estimator_name == "PoissonRegressor":
        return "poisson"
    if estimator_name == "GammaRegressor":
        return "gamma"
    return "tweedie"


def witness_glm_estimator_optimizer(estimator_name: str, solver: str) -> str:
    """Describe the optimizer boundary selected by a GLM estimator."""
    del estimator_name
    return solver


def witness_glm_estimator_methods(estimator_name: str) -> tuple[str, ...]:
    """Describe public methods exposed for framework-level GLM routing."""
    return (estimator_name,)


def witness_glm_fit_method_payload(
    estimator: object,
    X: object,
    y: object,
    *,
    sample_weight: object = None,
) -> dict[str, object]:
    """Describe a public GLM fit-method callback payload."""
    kwargs = {} if sample_weight is None else {"sample_weight": sample_weight}
    return {"estimator": estimator, "method_name": "fit", "args": (X, y), "kwargs": kwargs}


def witness_glm_prediction_method_payload(estimator: object, method_name: str, X: object) -> dict[str, object]:
    """Describe a public GLM prediction-like method callback payload."""
    return {"estimator": estimator, "method_name": method_name, "args": (X,), "kwargs": {}}


def witness_glm_fit_return_self(estimator: object) -> object:
    """Describe public GLM fit returning the fitted estimator object."""
    return estimator


def witness_glm_fitted_state_summary(estimator: object) -> dict[str, object]:
    """Describe fitted coefficient state exposed after GLM optimization."""
    return {"estimator": estimator}

