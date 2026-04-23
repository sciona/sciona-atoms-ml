from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.gaussian_process._gpc import _BinaryGaussianProcessClassifierLaplace
from sklearn.gaussian_process.kernels import ConstantKernel, RBF


def _single_step_model() -> tuple[_BinaryGaussianProcessClassifierLaplace, np.ndarray]:
    X = np.array([[-1.0], [-0.3], [0.2], [0.9], [1.4]], dtype=np.float64)
    y = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    kernel = ConstantKernel(1.3, constant_value_bounds="fixed") * RBF(0.8, length_scale_bounds="fixed")
    model = _BinaryGaussianProcessClassifierLaplace(
        kernel=kernel,
        optimizer=None,
        max_iter_predict=1,
        warm_start=False,
    )
    model.fit(X, y)
    return model, X


def _predictive_model() -> tuple[_BinaryGaussianProcessClassifierLaplace, np.ndarray]:
    X = np.array([[-1.0], [-0.3], [0.2], [0.9], [1.4]], dtype=np.float64)
    y = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    kernel = ConstantKernel(1.1, constant_value_bounds="fixed") * RBF(0.9, length_scale_bounds="fixed")
    model = _BinaryGaussianProcessClassifierLaplace(
        kernel=kernel,
        optimizer=None,
        max_iter_predict=25,
        warm_start=False,
    )
    model.fit(X, y)
    X_test = np.array([[-0.7], [0.0], [1.1]], dtype=np.float64)
    return model, X_test


def test_gp_classifier_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification import (
        gp_classifier_laplace_log_marginal_likelihood,
        gp_classifier_laplace_newton_step,
        gp_classifier_posterior_cross_solve,
        gp_classifier_posterior_mean,
        gp_classifier_posterior_variance,
        gp_classifier_predictive_proba,
        gp_classifier_predictive_probability,
    )

    assert callable(gp_classifier_laplace_log_marginal_likelihood)
    assert callable(gp_classifier_laplace_newton_step)
    assert callable(gp_classifier_posterior_cross_solve)
    assert callable(gp_classifier_posterior_mean)
    assert callable(gp_classifier_posterior_variance)
    assert callable(gp_classifier_predictive_proba)
    assert callable(gp_classifier_predictive_probability)


def test_gp_classifier_laplace_newton_step_matches_private_single_iteration() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification import (
        gp_classifier_laplace_log_marginal_likelihood,
        gp_classifier_laplace_newton_step,
    )

    model, X = _single_step_model()
    K = model.kernel_(X)
    initial_f = np.zeros_like(model.y_train_, dtype=np.float64)
    f_next, pi, W_sr, L, b, a = gp_classifier_laplace_newton_step(K, model.y_train_, initial_f)
    expected_lml, (expected_pi, expected_W_sr, expected_L, expected_b, expected_a) = model._posterior_mode(
        K,
        return_temporaries=True,
    )
    actual_lml = gp_classifier_laplace_log_marginal_likelihood(model.y_train_, f_next, a, L)

    assert np.allclose(f_next, model.f_cached)
    assert np.allclose(pi, expected_pi)
    assert np.allclose(W_sr, expected_W_sr)
    assert np.allclose(L, expected_L)
    assert np.allclose(b, expected_b)
    assert np.allclose(a, expected_a)
    assert np.isclose(actual_lml, expected_lml)


def test_gp_classifier_predictive_chain_matches_private_predict_proba() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification import (
        gp_classifier_posterior_cross_solve,
        gp_classifier_posterior_mean,
        gp_classifier_posterior_variance,
        gp_classifier_predictive_proba,
        gp_classifier_predictive_probability,
    )

    model, X_test = _predictive_model()
    K_star = model.kernel_(model.X_train_, X_test)
    f_star = gp_classifier_posterior_mean(K_star, model.y_train_, model.pi_)
    v = gp_classifier_posterior_cross_solve(model.L_, model.W_sr_, K_star)
    var_f_star = gp_classifier_posterior_variance(model.kernel_.diag(X_test), v)
    pi_star = gp_classifier_predictive_probability(f_star, var_f_star)
    proba = gp_classifier_predictive_proba(pi_star)

    expected = model.predict_proba(X_test)

    assert np.allclose(f_star, K_star.T.dot(model.y_train_ - model.pi_))
    assert np.allclose(proba, expected)


def test_gp_classifier_predictive_probability_respects_binary_stack() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification import (
        gp_classifier_predictive_proba,
        gp_classifier_predictive_probability,
    )

    f_star = np.array([-0.4, 0.2, 0.9], dtype=np.float64)
    var_f_star = np.array([0.3, 0.6, 0.5], dtype=np.float64)
    pi_star = gp_classifier_predictive_probability(f_star, var_f_star)
    proba = gp_classifier_predictive_proba(pi_star)

    assert np.all(pi_star >= 0.0)
    assert np.all(pi_star <= 1.0)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.allclose(proba[:, 1], pi_star)


def test_gp_classifier_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification import (
        gp_classifier_laplace_newton_step,
        gp_classifier_posterior_cross_solve,
        gp_classifier_posterior_mean,
        gp_classifier_predictive_probability,
    )

    model, X_test = _predictive_model()
    K = model.kernel_(model.X_train_)
    K_star = model.kernel_(model.X_train_, X_test)

    with pytest.raises(ViolationError):
        gp_classifier_laplace_newton_step(K, np.array([0.0, 2.0, 1.0, 0.0, 1.0]), np.zeros(5, dtype=np.float64))

    with pytest.raises(ViolationError):
        gp_classifier_laplace_newton_step(K, model.y_train_, np.zeros(4, dtype=np.float64))

    with pytest.raises(ViolationError):
        gp_classifier_posterior_mean(K_star, model.y_train_[:-1], model.pi_)

    with pytest.raises(ViolationError):
        gp_classifier_posterior_cross_solve(model.L_, model.W_sr_[:-1], K_star)

    with pytest.raises(ViolationError):
        gp_classifier_predictive_probability(np.array([0.1, 0.2], dtype=np.float64), np.array([0.0, 0.3], dtype=np.float64))
