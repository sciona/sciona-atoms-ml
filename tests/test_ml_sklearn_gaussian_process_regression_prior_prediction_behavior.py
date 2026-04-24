from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF


def _query_points() -> np.ndarray:
    return np.array([[-1.0], [-0.2], [0.4], [1.1]], dtype=np.float64)


def _fixed_kernel() -> ConstantKernel:
    return ConstantKernel(1.4, constant_value_bounds="fixed") * RBF(
        0.9, length_scale_bounds="fixed"
    )


def test_gp_regression_prior_prediction_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_prior_prediction import (
        gp_regression_prior_covariance,
        gp_regression_prior_mean,
        gp_regression_prior_std,
        gp_regression_prior_target_count,
        gp_regression_prior_variance,
    )

    assert callable(gp_regression_prior_target_count)
    assert callable(gp_regression_prior_mean)
    assert callable(gp_regression_prior_covariance)
    assert callable(gp_regression_prior_variance)
    assert callable(gp_regression_prior_std)


def test_gp_regression_prior_target_count_and_mean_match_unfitted_predict() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_prior_prediction import (
        gp_regression_prior_mean,
        gp_regression_prior_target_count,
    )

    X = _query_points()
    single_model = GaussianProcessRegressor(kernel=_fixed_kernel(), optimizer=None)
    multi_model = GaussianProcessRegressor(kernel=_fixed_kernel(), optimizer=None, n_targets=2)

    single_n_targets = gp_regression_prior_target_count(n_targets=single_model.n_targets)
    multi_n_targets = gp_regression_prior_target_count(n_targets=multi_model.n_targets)

    single_mean = gp_regression_prior_mean(X.shape[0], n_targets=single_n_targets)
    multi_mean = gp_regression_prior_mean(X.shape[0], n_targets=multi_n_targets)

    assert single_n_targets == 1
    assert multi_n_targets == 2
    assert np.array_equal(single_mean, single_model.predict(X))
    assert np.array_equal(multi_mean, multi_model.predict(X))


def test_gp_regression_prior_covariance_matches_unfitted_predict() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_prior_prediction import (
        gp_regression_prior_covariance,
        gp_regression_prior_target_count,
    )

    X = _query_points()
    kernel = _fixed_kernel()
    single_model = GaussianProcessRegressor(kernel=kernel, optimizer=None)
    multi_model = GaussianProcessRegressor(kernel=kernel, optimizer=None, n_targets=3)
    kernel_covariance = kernel(X)

    single_covariance = gp_regression_prior_covariance(
        kernel_covariance,
        n_targets=gp_regression_prior_target_count(n_targets=single_model.n_targets),
    )
    multi_covariance = gp_regression_prior_covariance(
        kernel_covariance,
        n_targets=gp_regression_prior_target_count(n_targets=multi_model.n_targets),
    )

    _, single_expected = single_model.predict(X, return_cov=True)
    _, multi_expected = multi_model.predict(X, return_cov=True)

    assert np.allclose(single_covariance, single_expected)
    assert np.allclose(multi_covariance, multi_expected)


def test_gp_regression_prior_variance_and_std_match_unfitted_predict() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_prior_prediction import (
        gp_regression_prior_std,
        gp_regression_prior_target_count,
        gp_regression_prior_variance,
    )

    X = _query_points()
    kernel = _fixed_kernel()
    single_model = GaussianProcessRegressor(kernel=kernel, optimizer=None)
    multi_model = GaussianProcessRegressor(kernel=kernel, optimizer=None, n_targets=2)
    kernel_variance = kernel.diag(X)

    single_variance = gp_regression_prior_variance(
        kernel_variance,
        n_targets=gp_regression_prior_target_count(n_targets=single_model.n_targets),
    )
    multi_variance = gp_regression_prior_variance(
        kernel_variance,
        n_targets=gp_regression_prior_target_count(n_targets=multi_model.n_targets),
    )

    single_std = gp_regression_prior_std(single_variance)
    multi_std = gp_regression_prior_std(multi_variance)

    _, single_expected = single_model.predict(X, return_std=True)
    _, multi_expected = multi_model.predict(X, return_std=True)

    assert np.allclose(single_std, single_expected)
    assert np.allclose(multi_std, multi_expected)


def test_gp_regression_prior_prediction_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_prior_prediction import (
        gp_regression_prior_covariance,
        gp_regression_prior_mean,
        gp_regression_prior_std,
        gp_regression_prior_target_count,
        gp_regression_prior_variance,
    )

    with pytest.raises(ViolationError):
        gp_regression_prior_target_count(n_targets=0)

    with pytest.raises(ViolationError):
        gp_regression_prior_mean(0, n_targets=1)

    with pytest.raises(ViolationError):
        gp_regression_prior_covariance(np.ones((2, 3), dtype=np.float64), n_targets=1)

    with pytest.raises(ViolationError):
        gp_regression_prior_variance(np.array([], dtype=np.float64), n_targets=1)

    with pytest.raises(ViolationError):
        gp_regression_prior_std(np.array([-1.0, 0.5], dtype=np.float64))
