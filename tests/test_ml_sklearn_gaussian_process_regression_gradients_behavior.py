from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF


def _single_output_model() -> GaussianProcessRegressor:
    X = np.array([[-1.0], [-0.2], [0.4], [1.1]], dtype=np.float64)
    y = np.array([0.3, -0.1, 0.7, 1.2], dtype=np.float64)
    kernel = ConstantKernel(1.4, constant_value_bounds=(1e-3, 10.0)) * RBF(0.9, length_scale_bounds=(1e-3, 10.0))
    model = GaussianProcessRegressor(kernel=kernel, alpha=0.05, optimizer=None, normalize_y=False)
    model.fit(X, y)
    return model


def _multi_output_model() -> GaussianProcessRegressor:
    X = np.array([[-1.0], [-0.2], [0.4], [1.1]], dtype=np.float64)
    y = np.column_stack(
        [
            np.array([0.3, -0.1, 0.7, 1.2], dtype=np.float64),
            np.array([1.0, 0.4, -0.2, 0.5], dtype=np.float64),
        ]
    )
    kernel = ConstantKernel(1.2, constant_value_bounds=(1e-3, 10.0)) * RBF(0.8, length_scale_bounds=(1e-3, 10.0))
    model = GaussianProcessRegressor(kernel=kernel, alpha=np.full(X.shape[0], 0.03), optimizer=None, normalize_y=True)
    model.fit(X, y)
    return model


def test_gp_regression_gradient_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_gradients import (
        gp_log_marginal_gradient,
        gp_log_marginal_gradient_dims,
        gp_log_marginal_gradient_inner_term,
    )

    assert callable(gp_log_marginal_gradient_inner_term)
    assert callable(gp_log_marginal_gradient_dims)
    assert callable(gp_log_marginal_gradient)


def test_gp_log_marginal_gradient_matches_sklearn_single_output() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_gradients import (
        gp_log_marginal_gradient,
        gp_log_marginal_gradient_dims,
        gp_log_marginal_gradient_inner_term,
    )

    model = _single_output_model()
    expected_lml, expected_gradient = model.log_marginal_likelihood(model.kernel_.theta, eval_gradient=True, clone_kernel=False)
    del expected_lml
    K, K_gradient = model.kernel_(model.X_train_, eval_gradient=True)
    del K

    inner_term = gp_log_marginal_gradient_inner_term(model.L_, model.alpha_)
    gradient_dims = gp_log_marginal_gradient_dims(inner_term, K_gradient)
    gradient = gp_log_marginal_gradient(gradient_dims)

    assert inner_term.shape == (model.X_train_.shape[0], model.X_train_.shape[0], 1)
    assert gradient_dims.shape == (expected_gradient.shape[0], 1)
    assert np.allclose(gradient, expected_gradient)


def test_gp_log_marginal_gradient_matches_sklearn_multi_output() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_gradients import (
        gp_log_marginal_gradient,
        gp_log_marginal_gradient_dims,
        gp_log_marginal_gradient_inner_term,
    )

    model = _multi_output_model()
    expected_lml, expected_gradient = model.log_marginal_likelihood(model.kernel_.theta, eval_gradient=True, clone_kernel=False)
    del expected_lml
    K, K_gradient = model.kernel_(model.X_train_, eval_gradient=True)
    del K

    inner_term = gp_log_marginal_gradient_inner_term(model.L_, model.alpha_)
    gradient_dims = gp_log_marginal_gradient_dims(inner_term, K_gradient)
    gradient = gp_log_marginal_gradient(gradient_dims)

    assert inner_term.shape == (model.X_train_.shape[0], model.X_train_.shape[0], model.y_train_.shape[1])
    assert gradient_dims.shape == (expected_gradient.shape[0], model.y_train_.shape[1])
    assert np.allclose(gradient, expected_gradient)


def test_gp_log_marginal_gradient_sum_is_identity_for_single_output_dims() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_gradients import gp_log_marginal_gradient

    dims = np.array([1.0, -2.0, 3.0], dtype=np.float64)
    assert np.array_equal(gp_log_marginal_gradient(dims), dims)


def test_contracts_reject_invalid_gp_gradient_inputs() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_gradients import (
        gp_log_marginal_gradient,
        gp_log_marginal_gradient_dims,
        gp_log_marginal_gradient_inner_term,
    )

    model = _single_output_model()
    _, K_gradient = model.kernel_(model.X_train_, eval_gradient=True)

    with pytest.raises(ViolationError):
        gp_log_marginal_gradient_inner_term(np.eye(3, dtype=np.float64), model.alpha_)

    with pytest.raises(ViolationError):
        gp_log_marginal_gradient_dims(
            np.ones((4, 4, 1), dtype=np.float64),
            K_gradient[:, :-1, :],
        )

    with pytest.raises(ViolationError):
        gp_log_marginal_gradient(np.ones((2, 0), dtype=np.float64))
