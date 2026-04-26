from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics.pairwise import (
    cosine_similarity,
    laplacian_kernel,
    linear_kernel,
    polynomial_kernel,
    sigmoid_kernel,
)


def _data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[0.0, 1.0], [2.0, 3.0], [1.5, -0.5], [0.0, 0.0]], dtype=np.float64)
    Y = np.array([[1.0, 1.0], [-1.0, 2.0], [0.0, 0.0]], dtype=np.float64)
    return X, Y


def test_pairwise_kernel_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.metrics_pairwise_kernels import (
        pairwise_cosine_similarity,
        pairwise_default_gamma,
        pairwise_laplacian_kernel,
        pairwise_linear_kernel,
        pairwise_polynomial_kernel,
        pairwise_sigmoid_kernel,
    )

    assert callable(pairwise_default_gamma)
    assert callable(pairwise_linear_kernel)
    assert callable(pairwise_polynomial_kernel)
    assert callable(pairwise_laplacian_kernel)
    assert callable(pairwise_sigmoid_kernel)
    assert callable(pairwise_cosine_similarity)


def test_pairwise_default_gamma_matches_sklearn_fallback() -> None:
    from sciona.atoms.ml.sklearn.metrics_pairwise_kernels import pairwise_default_gamma

    assert pairwise_default_gamma(4) == 0.25
    assert pairwise_default_gamma(4, 0.7) == 0.7


def test_pairwise_linear_kernel_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.metrics_pairwise_kernels import pairwise_linear_kernel

    X, Y = _data()
    assert np.allclose(pairwise_linear_kernel(X), linear_kernel(X))
    assert np.allclose(pairwise_linear_kernel(X, Y), linear_kernel(X, Y))


def test_pairwise_polynomial_kernel_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.metrics_pairwise_kernels import pairwise_polynomial_kernel

    X, Y = _data()
    assert np.allclose(pairwise_polynomial_kernel(X, degree=2.0, gamma=None, coef0=1.5), polynomial_kernel(X, degree=2.0, gamma=None, coef0=1.5))
    assert np.allclose(pairwise_polynomial_kernel(X, Y, degree=4.0, gamma=0.3, coef0=-0.25), polynomial_kernel(X, Y, degree=4.0, gamma=0.3, coef0=-0.25))


def test_pairwise_laplacian_kernel_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.metrics_pairwise_kernels import pairwise_laplacian_kernel

    X, Y = _data()
    assert np.allclose(pairwise_laplacian_kernel(X, gamma=None), laplacian_kernel(X, gamma=None))
    assert np.allclose(pairwise_laplacian_kernel(X, Y, gamma=0.4), laplacian_kernel(X, Y, gamma=0.4))


def test_pairwise_sigmoid_kernel_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.metrics_pairwise_kernels import pairwise_sigmoid_kernel

    X, Y = _data()
    assert np.allclose(pairwise_sigmoid_kernel(X, gamma=None, coef0=1.0), sigmoid_kernel(X, gamma=None, coef0=1.0))
    assert np.allclose(pairwise_sigmoid_kernel(X, Y, gamma=0.4, coef0=-0.2), sigmoid_kernel(X, Y, gamma=0.4, coef0=-0.2))


def test_pairwise_cosine_similarity_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.metrics_pairwise_kernels import pairwise_cosine_similarity

    X, Y = _data()
    assert np.allclose(pairwise_cosine_similarity(X), cosine_similarity(X))
    assert np.allclose(pairwise_cosine_similarity(X, Y), cosine_similarity(X, Y))


def test_pairwise_kernels_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.metrics_pairwise_kernels import (
        pairwise_cosine_similarity,
        pairwise_default_gamma,
        pairwise_laplacian_kernel,
        pairwise_linear_kernel,
        pairwise_polynomial_kernel,
        pairwise_sigmoid_kernel,
    )

    X, Y = _data()
    with pytest.raises(Exception):
        pairwise_default_gamma(0)
    with pytest.raises(Exception):
        pairwise_linear_kernel(X, Y[:, :1])
    with pytest.raises(Exception):
        pairwise_polynomial_kernel(X, degree=0.5)
    with pytest.raises(Exception):
        pairwise_laplacian_kernel(X, gamma=0.0)
    with pytest.raises(Exception):
        pairwise_sigmoid_kernel(X, gamma=-1.0)
    with pytest.raises(Exception):
        pairwise_cosine_similarity(X, Y[:, :1])
