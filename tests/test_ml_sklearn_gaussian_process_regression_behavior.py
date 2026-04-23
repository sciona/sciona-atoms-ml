from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF


def _single_output_model() -> tuple[GaussianProcessRegressor, np.ndarray]:
    X = np.array([[-1.0], [-0.2], [0.4], [1.1]], dtype=np.float64)
    y = np.array([0.3, -0.1, 0.7, 1.2], dtype=np.float64)
    kernel = ConstantKernel(1.4, constant_value_bounds="fixed") * RBF(0.9, length_scale_bounds="fixed")
    model = GaussianProcessRegressor(kernel=kernel, alpha=0.05, optimizer=None, normalize_y=False)
    model.fit(X, y)
    X_test = np.array([[-0.7], [0.0], [0.8]], dtype=np.float64)
    return model, X_test


def _multi_output_model() -> tuple[GaussianProcessRegressor, np.ndarray]:
    X = np.array([[-1.0], [-0.2], [0.4], [1.1]], dtype=np.float64)
    y = np.column_stack(
        [
            np.array([0.3, -0.1, 0.7, 1.2], dtype=np.float64),
            np.array([1.0, 0.4, -0.2, 0.5], dtype=np.float64),
        ]
    )
    kernel = ConstantKernel(1.2, constant_value_bounds="fixed") * RBF(0.8, length_scale_bounds="fixed")
    model = GaussianProcessRegressor(kernel=kernel, alpha=np.full(X.shape[0], 0.03), optimizer=None, normalize_y=True)
    model.fit(X, y)
    X_test = np.array([[-0.6], [0.2], [0.9]], dtype=np.float64)
    return model, X_test


def test_gp_regression_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression import (
        gp_dual_coefficients,
        gp_log_marginal_likelihood,
        gp_posterior_cross_solve,
        gp_posterior_predictive_covariance,
        gp_posterior_predictive_mean,
        gp_posterior_predictive_std,
        gp_regularized_train_kernel,
        gp_train_cholesky,
    )

    assert callable(gp_dual_coefficients)
    assert callable(gp_log_marginal_likelihood)
    assert callable(gp_posterior_cross_solve)
    assert callable(gp_posterior_predictive_covariance)
    assert callable(gp_posterior_predictive_mean)
    assert callable(gp_posterior_predictive_std)
    assert callable(gp_regularized_train_kernel)
    assert callable(gp_train_cholesky)


def test_gp_training_linear_algebra_matches_sklearn_single_output() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression import (
        gp_dual_coefficients,
        gp_log_marginal_likelihood,
        gp_regularized_train_kernel,
        gp_train_cholesky,
    )

    model, _ = _single_output_model()
    K = model.kernel_(model.X_train_)
    regularized = gp_regularized_train_kernel(K, model.alpha)
    L = gp_train_cholesky(regularized)
    dual = gp_dual_coefficients(L, model.y_train_)
    lml = gp_log_marginal_likelihood(model.y_train_, dual, L)

    assert np.allclose(regularized, K + np.eye(K.shape[0]) * model.alpha)
    assert np.allclose(L, model.L_)
    assert np.allclose(dual, model.alpha_)
    assert np.isclose(lml, model.log_marginal_likelihood_value_)


def test_gp_posterior_prediction_matches_sklearn_single_output() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression import (
        gp_posterior_cross_solve,
        gp_posterior_predictive_covariance,
        gp_posterior_predictive_mean,
        gp_posterior_predictive_std,
    )

    model, X_test = _single_output_model()
    K_trans = model.kernel_(X_test, model.X_train_)
    K_test = model.kernel_(X_test)
    V = gp_posterior_cross_solve(model.L_, K_trans)

    mean = gp_posterior_predictive_mean(K_trans, model.alpha_, model._y_train_mean, model._y_train_std)
    covariance = gp_posterior_predictive_covariance(K_test, V, model._y_train_std)
    std = gp_posterior_predictive_std(model.kernel_.diag(X_test), V, model._y_train_std)

    expected_mean = model.predict(X_test)
    expected_mean_cov, expected_covariance = model.predict(X_test, return_cov=True)
    expected_mean_std, expected_std = model.predict(X_test, return_std=True)

    assert np.allclose(mean, expected_mean)
    assert np.allclose(mean, expected_mean_cov)
    assert np.allclose(mean, expected_mean_std)
    assert np.allclose(covariance, expected_covariance)
    assert np.allclose(std, expected_std)


def test_gp_training_and_posterior_match_sklearn_multi_output() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression import (
        gp_dual_coefficients,
        gp_log_marginal_likelihood,
        gp_posterior_cross_solve,
        gp_posterior_predictive_covariance,
        gp_posterior_predictive_mean,
        gp_posterior_predictive_std,
        gp_regularized_train_kernel,
        gp_train_cholesky,
    )

    model, X_test = _multi_output_model()
    K = model.kernel_(model.X_train_)
    regularized = gp_regularized_train_kernel(K, model.alpha)
    L = gp_train_cholesky(regularized)
    dual = gp_dual_coefficients(L, model.y_train_)
    lml = gp_log_marginal_likelihood(model.y_train_, dual, L)
    K_trans = model.kernel_(X_test, model.X_train_)
    V = gp_posterior_cross_solve(L, K_trans)

    mean = gp_posterior_predictive_mean(K_trans, dual, model._y_train_mean, model._y_train_std)
    covariance = gp_posterior_predictive_covariance(model.kernel_(X_test), V, model._y_train_std)
    std = gp_posterior_predictive_std(model.kernel_.diag(X_test), V, model._y_train_std)

    expected_mean = model.predict(X_test)
    expected_mean_cov, expected_covariance = model.predict(X_test, return_cov=True)
    expected_mean_std, expected_std = model.predict(X_test, return_std=True)

    assert np.allclose(L, model.L_)
    assert np.allclose(dual, model.alpha_)
    assert np.isclose(lml, model.log_marginal_likelihood_value_)
    assert np.allclose(mean, expected_mean)
    assert np.allclose(mean, expected_mean_cov)
    assert np.allclose(mean, expected_mean_std)
    assert np.allclose(covariance, expected_covariance)
    assert np.allclose(std, expected_std)


def test_gp_posterior_std_clips_small_negative_variance() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression import gp_posterior_predictive_std

    kernel_diag = np.array([1.0], dtype=np.float64)
    V = np.array([[1.0 + 1e-12]], dtype=np.float64)
    std = gp_posterior_predictive_std(kernel_diag, V)

    assert np.array_equal(std, np.array([0.0]))


def test_contracts_reject_invalid_gp_inputs() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression import (
        gp_dual_coefficients,
        gp_posterior_cross_solve,
        gp_posterior_predictive_mean,
        gp_regularized_train_kernel,
        gp_train_cholesky,
    )

    model, X_test = _single_output_model()
    K_trans = model.kernel_(X_test, model.X_train_)

    with pytest.raises(ViolationError):
        gp_regularized_train_kernel(np.ones((2, 3), dtype=np.float64), 0.1)

    with pytest.raises(ViolationError):
        gp_regularized_train_kernel(model.kernel_(model.X_train_), np.array([0.1, 0.2], dtype=np.float64))

    with pytest.raises(ViolationError):
        gp_train_cholesky(np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float64))

    with pytest.raises(ViolationError):
        gp_dual_coefficients(model.L_, np.ones(model.L_.shape[0] + 1, dtype=np.float64))

    with pytest.raises(ViolationError):
        gp_posterior_cross_solve(model.L_, K_trans[:, :-1])

    with pytest.raises(ViolationError):
        gp_posterior_predictive_mean(K_trans, model.alpha_, y_train_std=0.0)
