from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics.pairwise import additive_chi2_kernel, chi2_kernel, rbf_kernel


def _data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[0.0, 1.0, 2.0], [2.0, 0.5, 1.5], [0.0, 0.0, 0.0]], dtype=np.float64)
    Y = np.array([[1.0, 1.0, 0.0], [0.5, 0.0, 3.0]], dtype=np.float64)
    return X, Y


def test_pairwise_distance_kernel_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.metrics_pairwise_distance_kernels import (
        pairwise_additive_chi2_kernel,
        pairwise_chi2_kernel,
        pairwise_rbf_kernel,
    )

    assert callable(pairwise_rbf_kernel)
    assert callable(pairwise_additive_chi2_kernel)
    assert callable(pairwise_chi2_kernel)


def test_pairwise_rbf_kernel_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.metrics_pairwise_distance_kernels import pairwise_rbf_kernel

    X, Y = _data()
    assert np.allclose(pairwise_rbf_kernel(X), rbf_kernel(X))
    assert np.allclose(pairwise_rbf_kernel(X, Y, gamma=0.4), rbf_kernel(X, Y, gamma=0.4))


def test_pairwise_additive_chi2_kernel_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.metrics_pairwise_distance_kernels import pairwise_additive_chi2_kernel

    X, Y = _data()
    assert np.allclose(pairwise_additive_chi2_kernel(X), additive_chi2_kernel(X))
    assert np.allclose(pairwise_additive_chi2_kernel(X, Y), additive_chi2_kernel(X, Y))


def test_pairwise_chi2_kernel_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.metrics_pairwise_distance_kernels import pairwise_chi2_kernel

    X, Y = _data()
    assert np.allclose(pairwise_chi2_kernel(X), chi2_kernel(X))
    assert np.allclose(pairwise_chi2_kernel(X, Y, gamma=0.4), chi2_kernel(X, Y, gamma=0.4))


def test_pairwise_distance_kernels_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.metrics_pairwise_distance_kernels import (
        pairwise_additive_chi2_kernel,
        pairwise_chi2_kernel,
        pairwise_rbf_kernel,
    )

    X, Y = _data()
    with pytest.raises(Exception):
        pairwise_rbf_kernel(X, gamma=0.0)
    with pytest.raises(Exception):
        pairwise_additive_chi2_kernel(X - 1.0, Y)
    with pytest.raises(Exception):
        pairwise_chi2_kernel(X, gamma=0.0)
