from __future__ import annotations

import numpy as np
import pytest
from sklearn.gaussian_process.kernels import ExpSineSquared, Matern, RationalQuadratic


def _data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[0.0, 1.0], [2.0, 3.0], [1.5, -0.5]], dtype=np.float64)
    Y = np.array([[1.0, 1.0], [-1.0, 2.0]], dtype=np.float64)
    return X, Y


def test_stationary_kernel_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.kernels import (
        exp_sine_squared_kernel,
        exp_sine_squared_kernel_diag,
        matern_kernel_diag,
        matern_kernel_matrix,
        rational_quadratic_kernel,
        rational_quadratic_kernel_diag,
    )

    assert callable(exp_sine_squared_kernel)
    assert callable(exp_sine_squared_kernel_diag)
    assert callable(matern_kernel_diag)
    assert callable(matern_kernel_matrix)
    assert callable(rational_quadratic_kernel)
    assert callable(rational_quadratic_kernel_diag)


def test_rational_quadratic_kernel_matches_sklearn_call_and_diag() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.kernels import rational_quadratic_kernel, rational_quadratic_kernel_diag

    X, Y = _data()
    expected = RationalQuadratic(length_scale=1.7, alpha=0.8)

    assert np.allclose(rational_quadratic_kernel(X, length_scale=1.7, alpha=0.8), expected(X))
    assert np.allclose(rational_quadratic_kernel(X, Y, length_scale=1.7, alpha=0.8), expected(X, Y))
    assert np.allclose(rational_quadratic_kernel_diag(X), expected.diag(X))


def test_matern_kernel_matches_sklearn_common_nu_values() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.kernels import matern_kernel_diag, matern_kernel_matrix

    X, Y = _data()
    for nu in (0.5, 1.5, 2.5, np.inf):
        expected = Matern(length_scale=1.2, nu=nu)
        assert np.allclose(matern_kernel_matrix(X, length_scale=1.2, nu=nu), expected(X))
        assert np.allclose(matern_kernel_matrix(X, Y, length_scale=1.2, nu=nu), expected(X, Y))
        assert np.allclose(matern_kernel_diag(X), expected.diag(X))


def test_matern_kernel_matches_sklearn_anisotropic_and_general_nu() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.kernels import matern_kernel_matrix

    X, Y = _data()
    expected_aniso = Matern(length_scale=np.array([1.2, 0.7], dtype=np.float64), nu=1.5)
    expected_general = Matern(length_scale=1.2, nu=0.9)

    assert np.allclose(matern_kernel_matrix(X, length_scale=(1.2, 0.7), nu=1.5), expected_aniso(X))
    assert np.allclose(matern_kernel_matrix(X, Y, length_scale=(1.2, 0.7), nu=1.5), expected_aniso(X, Y))
    assert np.allclose(matern_kernel_matrix(X, length_scale=1.2, nu=0.9), expected_general(X))
    assert np.allclose(matern_kernel_matrix(X, Y, length_scale=1.2, nu=0.9), expected_general(X, Y))


def test_exp_sine_squared_kernel_matches_sklearn_call_and_diag() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.kernels import exp_sine_squared_kernel, exp_sine_squared_kernel_diag

    X, Y = _data()
    expected = ExpSineSquared(length_scale=1.4, periodicity=2.2)

    assert np.allclose(exp_sine_squared_kernel(X, length_scale=1.4, periodicity=2.2), expected(X))
    assert np.allclose(exp_sine_squared_kernel(X, Y, length_scale=1.4, periodicity=2.2), expected(X, Y))
    assert np.allclose(exp_sine_squared_kernel_diag(X), expected.diag(X))


def test_stationary_kernels_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.kernels import (
        exp_sine_squared_kernel,
        matern_kernel_matrix,
        rational_quadratic_kernel,
    )

    X, Y = _data()
    with pytest.raises(Exception):
        rational_quadratic_kernel(X, length_scale=0.0)
    with pytest.raises(Exception):
        rational_quadratic_kernel(X, alpha=0.0)
    with pytest.raises(Exception):
        matern_kernel_matrix(X, length_scale=(1.0, 2.0, 3.0))
    with pytest.raises(Exception):
        matern_kernel_matrix(X, nu=0.0)
    with pytest.raises(Exception):
        exp_sine_squared_kernel(X, length_scale=0.0)
    with pytest.raises(Exception):
        exp_sine_squared_kernel(X, periodicity=0.0)
    with pytest.raises(Exception):
        matern_kernel_matrix(X, Y[:, :1], length_scale=1.0)
