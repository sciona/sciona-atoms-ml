from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.linear_model import GammaRegressor, PoissonRegressor, TweedieRegressor


def test_glm_fit_optimizer_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.glm_fit_optimizer_shell import (
        glm_fit_initial_coef,
        glm_fit_intercept_init_value,
        glm_fit_lbfgs_optimizer_payload,
        glm_fit_newton_solver_payload,
        glm_fit_result_attributes,
    )

    assert callable(glm_fit_initial_coef)
    assert callable(glm_fit_intercept_init_value)
    assert callable(glm_fit_lbfgs_optimizer_payload)
    assert callable(glm_fit_newton_solver_payload)
    assert callable(glm_fit_result_attributes)


def test_glm_fit_initial_coef_matches_cold_and_warm_start_layouts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.glm_fit_optimizer_shell import glm_fit_initial_coef

    cold_with_intercept = glm_fit_initial_coef(
        3,
        fit_intercept=True,
        warm_start=False,
        loss_dtype=np.float64,
        intercept_init=1.25,
    )
    cold_without_intercept = glm_fit_initial_coef(
        3,
        fit_intercept=False,
        warm_start=False,
        loss_dtype=np.float32,
        intercept_init=1.25,
    )
    warm_with_intercept = glm_fit_initial_coef(
        2,
        fit_intercept=True,
        warm_start=True,
        loss_dtype=np.float64,
        coef=np.array([0.4, -0.2], dtype=np.float64),
        intercept=0.7,
    )
    warm_without_intercept = glm_fit_initial_coef(
        2,
        fit_intercept=False,
        warm_start=True,
        loss_dtype=np.float32,
        coef=np.array([0.4, -0.2], dtype=np.float64),
        intercept=0.7,
    )

    np.testing.assert_array_equal(cold_with_intercept, np.array([0.0, 0.0, 0.0, 1.25], dtype=np.float64))
    assert cold_with_intercept.dtype == np.float64
    np.testing.assert_array_equal(cold_without_intercept, np.zeros(3, dtype=np.float32))
    assert cold_without_intercept.dtype == np.float32
    np.testing.assert_array_equal(warm_with_intercept, np.array([0.4, -0.2, 0.7], dtype=np.float64))
    np.testing.assert_array_equal(warm_without_intercept, np.array([0.4, -0.2], dtype=np.float32))


@pytest.mark.parametrize(
    ("estimator", "y", "sample_weight"),
    [
        (PoissonRegressor(), np.array([1.0, 2.0, 4.0], dtype=np.float64), None),
        (GammaRegressor(), np.array([1.0, 2.0, 4.0], dtype=np.float64), np.array([1.0, 2.0, 1.0], dtype=np.float64)),
        (TweedieRegressor(power=1.5), np.array([1.0, 2.0, 4.0], dtype=np.float64), np.array([2.0, 1.0, 1.0], dtype=np.float64)),
    ],
)
def test_glm_fit_intercept_init_value_matches_loss_link(
    estimator: PoissonRegressor,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
) -> None:
    from sciona.atoms.ml.sklearn.linear_model.glm_fit_optimizer_shell import glm_fit_intercept_init_value

    base_loss = estimator._get_loss()
    expected = base_loss.link.link(np.average(y, weights=sample_weight))

    result = glm_fit_intercept_init_value(base_loss, y, sample_weight=sample_weight)

    assert result == pytest.approx(expected)


def test_glm_fit_lbfgs_optimizer_payload_matches_sklearn_call_shape() -> None:
    from sciona.atoms.ml.sklearn.linear_model.glm import glm_dense_loss_gradient
    from sciona.atoms.ml.sklearn.linear_model.glm_fit_optimizer_shell import glm_fit_lbfgs_optimizer_payload

    X = np.array([[1.0, 2.0], [0.5, -1.0]], dtype=np.float64)
    y = np.array([1.5, 0.25], dtype=np.float64)
    coef = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    sample_weight = np.array([1.0, 2.0], dtype=np.float64)

    payload = glm_fit_lbfgs_optimizer_payload(
        glm_dense_loss_gradient,
        coef,
        X,
        y,
        sample_weight=sample_weight,
        l2_reg_strength=0.01,
        n_threads=3,
        max_iter=100,
        tol=1e-4,
        verbose=2,
    )

    assert payload["fun"] is glm_dense_loss_gradient
    assert payload["x0"] is coef
    assert payload["method"] == "L-BFGS-B"
    assert payload["jac"] is True
    assert payload["options"] == {
        "maxiter": 100,
        "maxls": 50,
        "iprint": 1,
        "gtol": 1e-4,
        "ftol": 64 * np.finfo(float).eps,
    }
    args = payload["args"]
    assert args[0] is X
    assert args[1] is y
    assert args[2] is sample_weight
    assert args[3] == 0.01
    assert args[4] == 3


def test_glm_fit_newton_solver_payload_matches_constructor_branches() -> None:
    from sciona.atoms.ml.sklearn.linear_model.glm_fit_optimizer_shell import glm_fit_newton_solver_payload

    class Solver:
        pass

    coef = np.array([0.1, -0.2], dtype=np.float64)
    linear_loss = object()

    cholesky_payload = glm_fit_newton_solver_payload(
        Solver,
        coef,
        linear_loss,
        l2_reg_strength=0.2,
        tol=1e-5,
        max_iter=25,
        n_threads=2,
        verbose=3,
    )
    hidden_payload = glm_fit_newton_solver_payload(
        Solver,
        coef,
        linear_loss,
        l2_reg_strength=0.2,
        tol=1e-5,
        max_iter=25,
        n_threads=2,
    )

    assert cholesky_payload == {
        "solver_class": Solver,
        "coef": coef,
        "linear_loss": linear_loss,
        "l2_reg_strength": 0.2,
        "tol": 1e-5,
        "max_iter": 25,
        "n_threads": 2,
        "verbose": 3,
    }
    assert hidden_payload == {
        "solver_class": Solver,
        "coef": coef,
        "linear_loss": linear_loss,
        "l2_reg_strength": 0.2,
        "tol": 1e-5,
        "max_iter": 25,
        "n_threads": 2,
    }


def test_glm_fit_result_attributes_match_fit_tail_unpacking() -> None:
    from sciona.atoms.ml.sklearn.linear_model.glm_fit_optimizer_shell import glm_fit_result_attributes

    with_intercept = glm_fit_result_attributes(np.array([0.4, -0.8, 0.15], dtype=np.float64), fit_intercept=True)
    without_intercept = glm_fit_result_attributes(np.array([0.4, -0.8], dtype=np.float64), fit_intercept=False)

    np.testing.assert_array_equal(with_intercept["coef"], np.array([0.4, -0.8], dtype=np.float64))
    assert with_intercept["intercept"] == 0.15
    np.testing.assert_array_equal(without_intercept["coef"], np.array([0.4, -0.8], dtype=np.float64))
    assert without_intercept["intercept"] == 0.0


def test_glm_fit_optimizer_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.glm_fit_optimizer_shell import (
        glm_fit_initial_coef,
        glm_fit_intercept_init_value,
        glm_fit_lbfgs_optimizer_payload,
        glm_fit_newton_solver_payload,
        glm_fit_result_attributes,
    )

    with pytest.raises(ViolationError):
        glm_fit_initial_coef(0, fit_intercept=True, warm_start=False, loss_dtype=np.float64)

    with pytest.raises(ViolationError):
        glm_fit_initial_coef(2, fit_intercept=True, warm_start=True, loss_dtype=np.float64, coef=np.ones(3), intercept=0.0)

    with pytest.raises(ViolationError):
        glm_fit_intercept_init_value(PoissonRegressor()._get_loss(), np.array([1.0, np.nan]))

    with pytest.raises(ViolationError):
        glm_fit_lbfgs_optimizer_payload(
            object(),
            np.array([0.0, 1.0], dtype=np.float64),
            object(),
            object(),
            sample_weight=None,
            l2_reg_strength=-0.1,
            n_threads=1,
            max_iter=100,
            tol=1e-4,
            verbose=0,
        )

    with pytest.raises(ViolationError):
        glm_fit_newton_solver_payload(
            object(),
            np.array([0.0, 1.0], dtype=np.float64),
            object(),
            l2_reg_strength=0.0,
            tol=0.0,
            max_iter=100,
            n_threads=1,
        )

    with pytest.raises(ViolationError):
        glm_fit_result_attributes(np.array([0.4], dtype=np.float64), fit_intercept=True)
