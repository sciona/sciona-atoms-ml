"""Binary Gaussian-process classification optimizer-bookkeeping atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from sklearn.utils import check_random_state

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_gpc_restart_bounds,
    witness_gpc_restart_thetas,
    witness_gpc_select_best_optimum,
)

RandomStateLike = int | np.random.RandomState | None


def _real_bounds_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 2
        and array.shape[0] >= 1
        and array.shape[1] == 2
        and not np.isnan(array).any()
    )


def _finite_bounds_matrix(values: object) -> bool:
    return _real_bounds_matrix(values) and bool(np.isfinite(np.asarray(values, dtype=np.float64)).all())


def _finite_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _finite_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _nonnegative_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 0)


def _restart_bounds_valid(result: NDArray[np.float64], bounds: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    source = np.asarray(bounds, dtype=np.float64)
    return bool(values.shape == source.shape and np.array_equal(values, source))


def _restart_thetas_valid(
    result: NDArray[np.float64],
    bounds: NDArray[np.float64],
    n_restarts_optimizer: int,
) -> bool:
    values = np.asarray(result, dtype=np.float64)
    source = np.asarray(bounds, dtype=np.float64)
    return bool(
        values.shape == (int(n_restarts_optimizer), source.shape[0])
        and np.all(np.isfinite(values))
        and np.all(values > 0.0)
    )


def _objective_selection_inputs_valid(
    candidate_thetas: NDArray[np.float64],
    objective_values: NDArray[np.float64],
) -> bool:
    thetas = np.asarray(candidate_thetas, dtype=np.float64)
    objectives = np.asarray(objective_values, dtype=np.float64)
    return bool(
        _finite_matrix(thetas)
        and _finite_vector(objectives)
        and thetas.shape[0] == objectives.shape[0]
    )


def _selected_optimum_valid(
    result: tuple[NDArray[np.float64], float],
    candidate_thetas: NDArray[np.float64],
    objective_values: NDArray[np.float64],
) -> bool:
    if not isinstance(result, tuple) or len(result) != 2:
        return False
    best_theta, log_marginal_likelihood_value = result
    thetas = np.asarray(candidate_thetas, dtype=np.float64)
    objectives = np.asarray(objective_values, dtype=np.float64)
    theta = np.asarray(best_theta, dtype=np.float64)
    best_index = int(np.argmin(objectives))
    return bool(
        theta.shape == (thetas.shape[1],)
        and np.all(np.isfinite(theta))
        and np.array_equal(theta, thetas[best_index])
        and np.isfinite(float(log_marginal_likelihood_value))
        and float(log_marginal_likelihood_value) == float(-np.min(objectives))
    )


@register_atom(witness_gpc_restart_bounds)
@icontract.require(
    lambda bounds: _real_bounds_matrix(bounds),
    "bounds must be a nonempty numeric matrix with shape (n_params, 2) and no NaNs",
)
@icontract.require(
    lambda n_restarts_optimizer=0: _nonnegative_int(n_restarts_optimizer),
    "n_restarts_optimizer must be a nonnegative integer",
)
@icontract.ensure(
    lambda result, bounds: _restart_bounds_valid(result, bounds),
    "validated restart bounds must preserve the original bounds matrix",
)
def gpc_restart_bounds(
    bounds: NDArray[np.float64],
    *,
    n_restarts_optimizer: int = 0,
) -> NDArray[np.float64]:
    """Validate optimizer restart bounds the way binary GPC fitting does."""
    values = np.asarray(bounds, dtype=np.float64)
    if int(n_restarts_optimizer) > 0 and not np.isfinite(values).all():
        raise ValueError(
            "Multiple optimizer restarts (n_restarts_optimizer>0) "
            "requires that all bounds are finite."
        )
    return np.asarray(values, dtype=np.float64)


@register_atom(witness_gpc_restart_thetas)
@icontract.require(
    lambda bounds: _finite_bounds_matrix(bounds),
    "bounds must be a finite nonempty numeric matrix with shape (n_params, 2)",
)
@icontract.require(
    lambda n_restarts_optimizer: _nonnegative_int(n_restarts_optimizer),
    "n_restarts_optimizer must be a nonnegative integer",
)
@icontract.ensure(
    lambda result, bounds, n_restarts_optimizer: _restart_thetas_valid(result, bounds, n_restarts_optimizer),
    "restart thetas must be positive and finite with shape (n_restarts_optimizer, n_params)",
)
def gpc_restart_thetas(
    bounds: NDArray[np.float64],
    *,
    n_restarts_optimizer: int,
    random_state: RandomStateLike = 0,
) -> NDArray[np.float64]:
    """Draw binary GPC optimizer restart thetas by exponentiating log-space uniform draws."""
    values = np.asarray(bounds, dtype=np.float64)
    restart_count = int(n_restarts_optimizer)
    if restart_count == 0:
        return np.empty((0, values.shape[0]), dtype=np.float64)
    rng = check_random_state(random_state)
    draws = [np.exp(rng.uniform(values[:, 0], values[:, 1])) for _ in range(restart_count)]
    return np.asarray(draws, dtype=np.float64)


@register_atom(witness_gpc_select_best_optimum)
@icontract.require(
    lambda candidate_thetas, objective_values: _objective_selection_inputs_valid(candidate_thetas, objective_values),
    "candidate_thetas must be a finite nonempty matrix aligned with a finite nonempty objective_values vector",
)
@icontract.ensure(
    lambda result, candidate_thetas, objective_values: _selected_optimum_valid(result, candidate_thetas, objective_values),
    "selected optimum must return the theta at argmin(objective_values) and the negated minimum objective value",
)
def gpc_select_best_optimum(
    candidate_thetas: NDArray[np.float64],
    objective_values: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """Select the best binary GPC optimizer result from candidate objective values."""
    thetas = np.asarray(candidate_thetas, dtype=np.float64)
    objectives = np.asarray(objective_values, dtype=np.float64)
    best_index = int(np.argmin(objectives))
    return np.asarray(thetas[best_index], dtype=np.float64), float(-np.min(objectives))
