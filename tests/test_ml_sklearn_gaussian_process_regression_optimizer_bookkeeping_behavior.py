from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.utils import check_random_state


def _finite_bounds() -> np.ndarray:
    kernel = ConstantKernel(1.4, constant_value_bounds=(1e-3, 10.0)) * RBF(
        0.9, length_scale_bounds=(1e-2, 5.0)
    )
    return np.asarray(kernel.bounds, dtype=np.float64)


def _infinite_bounds() -> np.ndarray:
    kernel = ConstantKernel(1.4, constant_value_bounds=(1e-3, 10.0)) * RBF(
        0.9, length_scale_bounds=(1e-2, np.inf)
    )
    return np.asarray(kernel.bounds, dtype=np.float64)


def test_gp_regression_optimizer_bookkeeping_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_optimizer_bookkeeping import (
        gp_regression_restart_bounds,
        gp_regression_restart_thetas,
        gp_regression_select_best_optimum,
    )

    assert callable(gp_regression_restart_bounds)
    assert callable(gp_regression_restart_thetas)
    assert callable(gp_regression_select_best_optimum)


def test_gp_regression_restart_bounds_matches_sklearn_guard() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_optimizer_bookkeeping import (
        gp_regression_restart_bounds,
    )

    finite_bounds = _finite_bounds()
    infinite_bounds = _infinite_bounds()

    assert np.array_equal(
        gp_regression_restart_bounds(finite_bounds, n_restarts_optimizer=3),
        finite_bounds,
    )
    assert np.array_equal(
        gp_regression_restart_bounds(infinite_bounds, n_restarts_optimizer=0),
        infinite_bounds,
    )

    with pytest.raises(
        ValueError,
        match="Multiple optimizer restarts \\(n_restarts_optimizer>0\\) requires that all bounds are finite\\.",
    ):
        gp_regression_restart_bounds(infinite_bounds, n_restarts_optimizer=1)


def test_gp_regression_restart_thetas_match_sklearn_restart_loop() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_optimizer_bookkeeping import (
        gp_regression_restart_thetas,
    )

    bounds = _finite_bounds()
    rng = check_random_state(7)
    expected = np.vstack([rng.uniform(bounds[:, 0], bounds[:, 1]) for _ in range(3)])

    actual = gp_regression_restart_thetas(bounds, n_restarts_optimizer=3, random_state=7)
    empty = gp_regression_restart_thetas(bounds, n_restarts_optimizer=0, random_state=7)

    assert np.allclose(actual, expected)
    assert empty.shape == (0, bounds.shape[0])


def test_gp_regression_select_best_optimum_matches_sklearn_selection() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_optimizer_bookkeeping import (
        gp_regression_select_best_optimum,
    )

    candidate_thetas = np.array(
        [
            [0.5, -0.3],
            [0.1, 0.4],
            [-0.2, 0.9],
        ],
        dtype=np.float64,
    )
    objective_values = np.array([2.0, 1.25, 1.25], dtype=np.float64)

    expected_index = int(np.argmin(objective_values))
    best_theta, log_marginal_likelihood_value = gp_regression_select_best_optimum(
        candidate_thetas,
        objective_values,
    )

    assert np.array_equal(best_theta, candidate_thetas[expected_index])
    assert log_marginal_likelihood_value == float(-np.min(objective_values))


def test_gp_regression_optimizer_bookkeeping_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_optimizer_bookkeeping import (
        gp_regression_restart_bounds,
        gp_regression_restart_thetas,
        gp_regression_select_best_optimum,
    )

    with pytest.raises(ViolationError):
        gp_regression_restart_bounds(np.array([0.0, 1.0], dtype=np.float64), n_restarts_optimizer=1)

    with pytest.raises(ViolationError):
        gp_regression_restart_thetas(_infinite_bounds(), n_restarts_optimizer=1, random_state=0)

    with pytest.raises(ViolationError):
        gp_regression_select_best_optimum(
            np.ones((2, 2), dtype=np.float64),
            np.ones(3, dtype=np.float64),
        )
