"""Ghost witnesses for sklearn GLM fit optimizer-shell atoms."""

from __future__ import annotations

from collections.abc import Mapping


def witness_glm_fit_initial_coef(
    n_features: int,
    fit_intercept: bool,
    warm_start: bool,
    loss_dtype: object,
    coef: object,
    intercept: float,
    intercept_init: float,
) -> object:
    """Describe _GeneralizedLinearRegressor.fit initial coefficient setup."""
    return (n_features, fit_intercept, warm_start, loss_dtype, coef, intercept, intercept_init)


def witness_glm_fit_intercept_init_value(base_loss: object, y: object, sample_weight: object) -> float:
    """Describe the GLM cold-start intercept initialization value."""
    return float(base_loss.link.link(0.0))


def witness_glm_fit_lbfgs_optimizer_payload(
    objective: object,
    coef: object,
    X: object,
    y: object,
    sample_weight: object,
    l2_reg_strength: float,
    n_threads: int,
    max_iter: int,
    tol: float,
    verbose: int,
) -> Mapping[str, object]:
    """Describe the scipy.optimize.minimize payload assembled by GLM fit."""
    return {
        "fun": objective,
        "x0": coef,
        "args": (X, y, sample_weight, l2_reg_strength, n_threads),
        "max_iter": max_iter,
        "tol": tol,
        "verbose": verbose,
    }


def witness_glm_fit_newton_solver_payload(
    solver_class: object,
    coef: object,
    linear_loss: object,
    l2_reg_strength: float,
    tol: float,
    max_iter: int,
    n_threads: int,
    verbose: int | None,
) -> Mapping[str, object]:
    """Describe the Newton solver constructor payload assembled by GLM fit."""
    return {
        "solver_class": solver_class,
        "coef": coef,
        "linear_loss": linear_loss,
        "l2_reg_strength": l2_reg_strength,
        "tol": tol,
        "max_iter": max_iter,
        "n_threads": n_threads,
        "verbose": verbose,
    }


def witness_glm_fit_result_attributes(coef: object, fit_intercept: bool) -> Mapping[str, object]:
    """Describe final GLM coefficient/intercept attributes unpacked from fit."""
    return {"coef": coef, "fit_intercept": fit_intercept}
