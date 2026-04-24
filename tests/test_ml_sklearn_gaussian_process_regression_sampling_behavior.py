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


def test_gp_regression_sampling_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_sampling import (
        gp_sample_y_multi_output,
        gp_sample_y_single_output,
    )

    assert callable(gp_sample_y_single_output)
    assert callable(gp_sample_y_multi_output)


def test_gp_sample_y_single_output_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_sampling import gp_sample_y_single_output

    model, X_test = _single_output_model()
    y_mean, y_cov = model.predict(X_test, return_cov=True)

    actual = gp_sample_y_single_output(y_mean, y_cov, n_samples=4, random_state=17)
    expected = model.sample_y(X_test, n_samples=4, random_state=17)

    assert np.allclose(actual, expected)


def test_gp_sample_y_multi_output_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_sampling import gp_sample_y_multi_output

    model, X_test = _multi_output_model()
    y_mean, y_cov = model.predict(X_test, return_cov=True)

    actual = gp_sample_y_multi_output(y_mean, y_cov, n_samples=5, random_state=23)
    expected = model.sample_y(X_test, n_samples=5, random_state=23)

    assert np.allclose(actual, expected)


def test_gp_sample_y_single_output_shape_is_point_by_sample() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_sampling import gp_sample_y_single_output

    y_mean = np.array([0.0, 1.0], dtype=np.float64)
    y_cov = np.array([[1.0, 0.2], [0.2, 0.5]], dtype=np.float64)
    result = gp_sample_y_single_output(y_mean, y_cov, n_samples=3, random_state=0)
    assert result.shape == (2, 3)


def test_gp_sampling_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_sampling import (
        gp_sample_y_multi_output,
        gp_sample_y_single_output,
    )

    with pytest.raises(ViolationError):
        gp_sample_y_single_output(
            np.array([0.0, 1.0], dtype=np.float64),
            np.ones((2, 3), dtype=np.float64),
            n_samples=1,
            random_state=0,
        )

    with pytest.raises(ViolationError):
        gp_sample_y_single_output(
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
            n_samples=0,
            random_state=0,
        )

    with pytest.raises(ViolationError):
        gp_sample_y_multi_output(
            np.ones((3, 2), dtype=np.float64),
            np.ones((3, 3, 1), dtype=np.float64),
            n_samples=1,
            random_state=0,
        )
