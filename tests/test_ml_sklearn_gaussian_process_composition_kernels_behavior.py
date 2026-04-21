from __future__ import annotations

import numpy as np
import pytest
from sklearn.gaussian_process.kernels import CompoundKernel, ConstantKernel, Exponentiation, Product, RBF, Sum, WhiteKernel


def _data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[0.0, 1.0], [2.0, 3.0], [1.5, -0.5]], dtype=np.float64)
    Y = np.array([[1.0, 1.0], [-1.0, 2.0]], dtype=np.float64)
    return X, Y


def test_composition_kernel_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.kernels import (
        compound_kernel_diag_stack,
        compound_kernel_stack,
        exponentiation_kernel_diag,
        exponentiation_kernel_matrix,
        product_kernel_diag,
        product_kernel_matrix,
        sum_kernel_diag,
        sum_kernel_matrix,
    )

    assert callable(compound_kernel_diag_stack)
    assert callable(compound_kernel_stack)
    assert callable(exponentiation_kernel_diag)
    assert callable(exponentiation_kernel_matrix)
    assert callable(product_kernel_diag)
    assert callable(product_kernel_matrix)
    assert callable(sum_kernel_diag)
    assert callable(sum_kernel_matrix)


def test_sum_kernel_matrix_and_diag_match_sklearn_composition() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.kernels import (
        constant_kernel,
        constant_kernel_diag,
        rbf_kernel_diag,
        rbf_kernel_matrix,
        sum_kernel_diag,
        sum_kernel_matrix,
    )

    X, Y = _data()
    expected = Sum(ConstantKernel(constant_value=1.5), RBF(length_scale=1.7))

    K1 = constant_kernel(X, constant_value=1.5)
    K2 = rbf_kernel_matrix(X, length_scale=1.7)
    K1_cross = constant_kernel(X, Y, constant_value=1.5)
    K2_cross = rbf_kernel_matrix(X, Y, length_scale=1.7)

    assert np.allclose(sum_kernel_matrix(K1, K2), expected(X))
    assert np.allclose(sum_kernel_matrix(K1_cross, K2_cross), expected(X, Y))
    assert np.allclose(sum_kernel_diag(constant_kernel_diag(X, constant_value=1.5), rbf_kernel_diag(X)), expected.diag(X))


def test_product_kernel_matrix_and_diag_match_sklearn_composition() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.kernels import (
        constant_kernel,
        constant_kernel_diag,
        product_kernel_diag,
        product_kernel_matrix,
        rbf_kernel_diag,
        rbf_kernel_matrix,
    )

    X, Y = _data()
    expected = Product(ConstantKernel(constant_value=2.0), RBF(length_scale=0.8))

    K1 = constant_kernel(X, constant_value=2.0)
    K2 = rbf_kernel_matrix(X, length_scale=0.8)
    K1_cross = constant_kernel(X, Y, constant_value=2.0)
    K2_cross = rbf_kernel_matrix(X, Y, length_scale=0.8)

    assert np.allclose(product_kernel_matrix(K1, K2), expected(X))
    assert np.allclose(product_kernel_matrix(K1_cross, K2_cross), expected(X, Y))
    assert np.allclose(product_kernel_diag(constant_kernel_diag(X, constant_value=2.0), rbf_kernel_diag(X)), expected.diag(X))


def test_exponentiation_kernel_matrix_and_diag_match_sklearn_composition() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.kernels import (
        exponentiation_kernel_diag,
        exponentiation_kernel_matrix,
        rbf_kernel_diag,
        rbf_kernel_matrix,
    )

    X, Y = _data()
    expected = Exponentiation(RBF(length_scale=1.2), exponent=2.0)

    K = rbf_kernel_matrix(X, length_scale=1.2)
    K_cross = rbf_kernel_matrix(X, Y, length_scale=1.2)
    diag = rbf_kernel_diag(X)

    assert np.allclose(exponentiation_kernel_matrix(K, exponent=2.0), expected(X))
    assert np.allclose(exponentiation_kernel_matrix(K_cross, exponent=2.0), expected(X, Y))
    assert np.allclose(exponentiation_kernel_diag(diag, exponent=2.0), expected.diag(X))


def test_compound_kernel_stack_and_diag_match_sklearn_composition() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.kernels import (
        compound_kernel_diag_stack,
        compound_kernel_stack,
        rbf_kernel_diag,
        rbf_kernel_matrix,
        white_kernel,
        white_kernel_diag,
    )

    X, Y = _data()
    expected = CompoundKernel([WhiteKernel(noise_level=0.4), RBF(length_scale=1.6)])

    K_white = white_kernel(X, noise_level=0.4)
    K_rbf = rbf_kernel_matrix(X, length_scale=1.6)
    K_white_cross = white_kernel(X, Y, noise_level=0.4)
    K_rbf_cross = rbf_kernel_matrix(X, Y, length_scale=1.6)

    assert np.allclose(compound_kernel_stack((K_white, K_rbf)), expected(X))
    assert np.allclose(compound_kernel_stack((K_white_cross, K_rbf_cross)), expected(X, Y))
    assert np.allclose(compound_kernel_diag_stack((white_kernel_diag(X, noise_level=0.4), rbf_kernel_diag(X))), expected.diag(X))


def test_composition_kernels_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.kernels import (
        compound_kernel_diag_stack,
        compound_kernel_stack,
        exponentiation_kernel_matrix,
        product_kernel_diag,
        sum_kernel_matrix,
    )

    X, Y = _data()
    with pytest.raises(Exception):
        sum_kernel_matrix(X, Y)
    with pytest.raises(Exception):
        product_kernel_diag(np.ones(3, dtype=np.float64), np.ones(2, dtype=np.float64))
    with pytest.raises(Exception):
        exponentiation_kernel_matrix(X, exponent=-1.0)
    with pytest.raises(Exception):
        compound_kernel_stack(())
    with pytest.raises(Exception):
        compound_kernel_diag_stack((np.ones(3, dtype=np.float64), np.ones(2, dtype=np.float64)))
