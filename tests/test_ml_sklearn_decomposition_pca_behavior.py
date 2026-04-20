from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import PCA as SklearnPCA


def _data() -> np.ndarray:
    return np.array(
        [
            [0.0, 1.0, 3.0, 2.0],
            [1.0, 2.0, 4.0, 3.0],
            [2.0, 4.0, 7.0, 5.0],
            [4.0, 8.0, 9.0, 7.0],
            [5.0, 9.0, 12.0, 11.0],
            [6.0, 11.0, 13.0, 13.0],
        ],
        dtype=np.float64,
    )


def _assert_state_matches_sklearn(state, expected: SklearnPCA) -> None:
    assert np.allclose(state.components, expected.components_)
    assert np.allclose(state.explained_variance, expected.explained_variance_)
    assert np.allclose(state.explained_variance_ratio, expected.explained_variance_ratio_)
    assert np.allclose(state.singular_values, expected.singular_values_)
    assert np.allclose(state.mean, expected.mean_)
    assert np.isclose(state.noise_variance, expected.noise_variance_)
    assert state.n_samples == expected.n_samples_
    assert state.n_components == expected.n_components_
    assert state.n_features_in == expected.n_features_in_


def test_pca_fit_imports() -> None:
    from sciona.atoms.ml.sklearn.decomposition import PCAState, pca_fit

    assert PCAState is not None
    assert callable(pca_fit)


def test_pca_fit_matches_sklearn_full_solver_int_components() -> None:
    from sciona.atoms.ml.sklearn.decomposition import pca_fit

    X = _data()
    state = pca_fit(X, n_components=2, svd_solver="full")
    expected = SklearnPCA(n_components=2, svd_solver="full").fit(X)

    _assert_state_matches_sklearn(state, expected)
    assert state.svd_solver == "full"
    assert state.whiten is False


def test_pca_fit_matches_sklearn_full_solver_fractional_components() -> None:
    from sciona.atoms.ml.sklearn.decomposition import pca_fit

    X = _data()
    state = pca_fit(X, n_components=0.95, whiten=True)
    expected = SklearnPCA(n_components=0.95, whiten=True, svd_solver="full").fit(X)

    _assert_state_matches_sklearn(state, expected)
    assert state.whiten is True


def test_pca_fit_matches_sklearn_full_solver_default_components() -> None:
    from sciona.atoms.ml.sklearn.decomposition import pca_fit

    X = _data()
    state = pca_fit(X)
    expected = SklearnPCA(svd_solver="full").fit(X)

    _assert_state_matches_sklearn(state, expected)
    assert state.n_components == min(X.shape)


def test_pca_fit_rejects_unsupported_solver_and_constant_inputs() -> None:
    from sciona.atoms.ml.sklearn.decomposition import pca_fit

    X = _data()
    with pytest.raises(Exception):
        pca_fit(X, svd_solver="randomized")

    with pytest.raises(Exception):
        pca_fit(np.ones((4, 3), dtype=np.float64))
