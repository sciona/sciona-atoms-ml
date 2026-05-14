"""Ghost witnesses for sklearn HuberRegressor fit optimizer-shell atoms."""

from __future__ import annotations

from collections.abc import Mapping


def witness_huber_fit_initial_parameters(
    n_features: int,
    fit_intercept: bool,
    warm_start: bool,
    coef: object,
    intercept: float,
    scale: float,
) -> object:
    """Describe HuberRegressor's initial optimizer parameter vector."""
    return (n_features, fit_intercept, warm_start, coef, intercept, scale)


def witness_huber_fit_bounds(n_parameters: int) -> object:
    """Describe HuberRegressor's L-BFGS-B bounds matrix."""
    return n_parameters


def witness_huber_fit_optimizer_payload(
    objective: object,
    parameters: object,
    X: object,
    y: object,
    epsilon: float,
    alpha: float,
    sample_weight: object,
    max_iter: int,
    tol: float,
    bounds: object,
) -> Mapping[str, object]:
    """Describe HuberRegressor's optimize.minimize payload."""
    return {
        "fun": objective,
        "x0": parameters,
        "args": (X, y, epsilon, alpha, sample_weight),
        "max_iter": max_iter,
        "tol": tol,
        "bounds": bounds,
    }


def witness_huber_fit_status2_failure_message(status: int, message: object) -> str | None:
    """Describe the HuberRegressor status-2 convergence failure message."""
    return str(message) if status == 2 else None


def witness_huber_fit_result_attributes(parameters: object, n_features: int, fit_intercept: bool) -> Mapping[str, object]:
    """Describe fitted HuberRegressor attributes unpacked from optimizer parameters."""
    return {"parameters": parameters, "n_features": n_features, "fit_intercept": fit_intercept}


def witness_huber_fit_outlier_handoff_payload(
    X: object,
    y: object,
    coef: object,
    intercept: float,
    scale: float,
    epsilon: float,
) -> Mapping[str, object]:
    """Describe the postfit residual/outlier atom handoff payload."""
    return {"X": X, "y": y, "coef": coef, "intercept": intercept, "scale": scale, "epsilon": epsilon}
