from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import KernelPCA as SklearnKernelPCA


def _data() -> np.ndarray:
    return np.array(
        [
            [0.0, 1.0, 3.0],
            [1.0, 2.0, 4.0],
            [2.0, 0.0, 2.0],
            [4.0, 3.0, 1.0],
            [5.0, 5.0, 2.0],
        ],
        dtype=np.float64,
    )


def test_kernel_pca_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.decomposition import KernelPCAState, kernel_pca_fit, kernel_pca_transform

    assert KernelPCAState is not None
    assert callable(kernel_pca_fit)
    assert callable(kernel_pca_transform)


def test_kernel_pca_fit_and_transform_match_sklearn_dense_linear() -> None:
    from sciona.atoms.ml.sklearn.decomposition import kernel_pca_fit, kernel_pca_transform

    X = _data()
    state = kernel_pca_fit(X, n_components=2)
    expected = SklearnKernelPCA(
        n_components=2,
        kernel="linear",
        eigen_solver="dense",
        fit_inverse_transform=False,
        remove_zero_eig=False,
    ).fit(X)

    assert np.allclose(state.eigenvalues, expected.eigenvalues_)
    assert np.allclose(state.eigenvectors, expected.eigenvectors_)
    assert np.allclose(state.X_fit, expected.X_fit_)
    assert np.allclose(state.kernel_centerer_rows, expected._centerer.K_fit_rows_)
    assert np.isclose(state.kernel_centerer_all, expected._centerer.K_fit_all_)
    assert state.n_components == expected.eigenvalues_.shape[0]
    assert state.n_features_in == expected.n_features_in_

    query = X[[0, 2, 4]]
    assert np.allclose(kernel_pca_transform(query, state), expected.transform(query))


def test_kernel_pca_training_projection_matches_sklearn_fit_transform() -> None:
    from sciona.atoms.ml.sklearn.decomposition import kernel_pca_fit, kernel_pca_transform

    X = _data()
    state = kernel_pca_fit(X, n_components=3, gamma=0.5)
    expected = SklearnKernelPCA(n_components=3, kernel="linear", gamma=0.5, eigen_solver="dense").fit(X)

    assert state.gamma == expected.gamma_
    assert np.allclose(kernel_pca_transform(X, state), expected.fit_transform(X))


def test_kernel_pca_rejects_unsupported_options() -> None:
    from sciona.atoms.ml.sklearn.decomposition import kernel_pca_fit, kernel_pca_transform

    X = _data()
    with pytest.raises(Exception):
        kernel_pca_fit(X, kernel="rbf")

    with pytest.raises(Exception):
        kernel_pca_fit(X, eigen_solver="arpack")

    with pytest.raises(Exception):
        kernel_pca_fit(X, fit_inverse_transform=True)

    with pytest.raises(Exception):
        kernel_pca_fit(X, remove_zero_eig=True)

    with pytest.raises(Exception):
        kernel_pca_fit(np.ones((4, 3), dtype=np.float64))

    state = kernel_pca_fit(X, n_components=2)
    with pytest.raises(Exception):
        kernel_pca_transform(np.ones((2, 2), dtype=np.float64), state)
