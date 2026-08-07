"""Ghost witnesses for public Gaussian-process estimator API-shell atoms."""

from __future__ import annotations


def witness_gp_public_estimator_catalog(catalog_scope: str = "public_estimators") -> tuple[str, ...]:
    """Describe public Gaussian-process estimator names covered by this shell."""
    del catalog_scope
    return ("GaussianProcessRegressor", "GaussianProcessClassifier")


def witness_gp_public_estimator_task(estimator_name: str) -> str:
    """Describe the public Gaussian-process estimator task family."""
    return "regression" if estimator_name == "GaussianProcessRegressor" else "classification"


def witness_gp_public_optimizer_boundary(estimator_name: str, optimizer: object) -> str:
    """Describe the optimizer boundary behind a public Gaussian-process estimator."""
    del estimator_name
    if optimizer is None:
        return "no_optimizer"
    if optimizer == "fmin_l_bfgs_b":
        return "scipy_lbfgsb_optimizer"
    return "callable_optimizer"


def witness_gp_public_estimator_methods(estimator_name: str) -> tuple[str, ...]:
    """Describe public methods exposed for Gaussian-process routing."""
    return (estimator_name,)


def witness_gp_public_fit_method_payload(estimator: object, X: object, y: object) -> dict[str, object]:
    """Describe a public Gaussian-process fit callback payload."""
    return {"estimator": estimator, "method_name": "fit", "args": (X, y), "kwargs": {}}


def witness_gp_public_prediction_method_payload(
    estimator: object,
    method_name: str,
    X: object,
    *,
    return_std: bool = False,
    return_cov: bool = False,
    n_samples: int = 1,
    random_state: int | None = 0,
) -> dict[str, object]:
    """Describe a public Gaussian-process prediction-like callback payload."""
    del return_std, return_cov, n_samples, random_state
    return {"estimator": estimator, "method_name": method_name, "args": (X,), "kwargs": {}}


def witness_gp_public_fit_return_self(estimator: object) -> object:
    """Describe public Gaussian-process fit returning the fitted estimator."""
    return estimator


def witness_gp_public_fitted_state_summary(estimator: object) -> dict[str, object]:
    """Describe fitted Gaussian-process state after delegated execution."""
    return {"estimator": estimator}
