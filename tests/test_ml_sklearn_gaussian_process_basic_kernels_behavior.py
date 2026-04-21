from __future__ import annotations

import numpy as np
import pytest
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, RBF, WhiteKernel


def _data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[0.0, 1.0], [2.0, 3.0], [1.5, -0.5]], dtype=np.float64)
    Y = np.array([[1.0, 1.0], [-1.0, 2.0]], dtype=np.float64)
    return X, Y


def test_basic_kernel_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.kernels import (
        constant_kernel,
        constant_kernel_diag,
        dot_product_kernel,
        dot_product_kernel_diag,
        rbf_kernel_diag,
        rbf_kernel_matrix,
        white_kernel,
        white_kernel_diag,
    )

    assert callable(constant_kernel)
    assert callable(constant_kernel_diag)
    assert callable(dot_product_kernel)
    assert callable(dot_product_kernel_diag)
    assert callable(rbf_kernel_diag)
    assert callable(rbf_kernel_matrix)
    assert callable(white_kernel)
    assert callable(white_kernel_diag)


def test_constant_kernel_matches_sklearn_call_and_diag() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.kernels import constant_kernel, constant_kernel_diag

    X, Y = _data()
    expected = ConstantKernel(constant_value=2.5)

    assert np.allclose(constant_kernel(X, constant_value=2.5), expected(X))
    assert np.allclose(constant_kernel(X, Y, constant_value=2.5), expected(X, Y))
    assert np.allclose(constant_kernel_diag(X, constant_value=2.5), expected.diag(X))


def test_white_kernel_matches_sklearn_call_and_diag() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.kernels import white_kernel, white_kernel_diag

    X, Y = _data()
    expected = WhiteKernel(noise_level=0.75)

    assert np.allclose(white_kernel(X, noise_level=0.75), expected(X))
    assert np.allclose(white_kernel(X, Y, noise_level=0.75), expected(X, Y))
    assert np.allclose(white_kernel_diag(X, noise_level=0.75), expected.diag(X))


def test_dot_product_kernel_matches_sklearn_call_and_diag() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.kernels import dot_product_kernel, dot_product_kernel_diag

    X, Y = _data()
    expected = DotProduct(sigma_0=1.25)

    assert np.allclose(dot_product_kernel(X, sigma_0=1.25), expected(X))
    assert np.allclose(dot_product_kernel(X, Y, sigma_0=1.25), expected(X, Y))
    assert np.allclose(dot_product_kernel_diag(X, sigma_0=1.25), expected.diag(X))


def test_rbf_kernel_matches_sklearn_call_and_diag() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.kernels import rbf_kernel_diag, rbf_kernel_matrix

    X, Y = _data()
    isotropic = RBF(length_scale=1.7)
    anisotropic = RBF(length_scale=np.array([1.7, 0.5], dtype=np.float64))

    assert np.allclose(rbf_kernel_matrix(X, length_scale=1.7), isotropic(X))
    assert np.allclose(rbf_kernel_matrix(X, Y, length_scale=1.7), isotropic(X, Y))
    assert np.allclose(rbf_kernel_matrix(X, length_scale=(1.7, 0.5)), anisotropic(X))
    assert np.allclose(rbf_kernel_matrix(X, Y, length_scale=(1.7, 0.5)), anisotropic(X, Y))
    assert np.allclose(rbf_kernel_diag(X), isotropic.diag(X))


def test_basic_kernels_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.kernels import (
        constant_kernel,
        dot_product_kernel,
        rbf_kernel_matrix,
        white_kernel,
    )

    X, Y = _data()
    with pytest.raises(Exception):
        constant_kernel(X, constant_value=0.0)
    with pytest.raises(Exception):
        white_kernel(X, noise_level=-1.0)
    with pytest.raises(Exception):
        dot_product_kernel(X, sigma_0=-1.0)
    with pytest.raises(Exception):
        rbf_kernel_matrix(X, length_scale=0.0)
    with pytest.raises(Exception):
        rbf_kernel_matrix(X, length_scale=(1.0, 2.0, 3.0))
    with pytest.raises(Exception):
        rbf_kernel_matrix(X, Y[:, :1], length_scale=1.0)
