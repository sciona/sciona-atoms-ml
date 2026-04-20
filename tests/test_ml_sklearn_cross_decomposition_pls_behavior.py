from __future__ import annotations

import numpy as np
import pytest
from sklearn.cross_decomposition import CCA as SklearnCCA
from sklearn.cross_decomposition import PLSCanonical as SklearnPLSCanonical
from sklearn.cross_decomposition import PLSRegression as SklearnPLSRegression


def _xy() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [0.0, 1.0, 3.0],
            [1.0, 2.0, 4.0],
            [2.0, 4.0, 7.0],
            [4.0, 8.0, 9.0],
            [5.0, 9.0, 12.0],
            [6.0, 11.0, 13.0],
        ],
        dtype=np.float64,
    )
    y = np.array(
        [
            [0.0, 1.0],
            [1.0, 1.5],
            [2.0, 2.5],
            [3.0, 4.0],
            [5.0, 6.0],
            [8.0, 9.0],
        ],
        dtype=np.float64,
    )
    return X, y


def _assert_common_state(state, expected) -> None:
    assert np.allclose(state.x_weights, expected.x_weights_)
    assert np.allclose(state.y_weights, expected.y_weights_)
    assert np.allclose(state.x_loadings, expected.x_loadings_)
    assert np.allclose(state.y_loadings, expected.y_loadings_)
    assert np.allclose(state.x_rotations, expected.x_rotations_)
    assert np.allclose(state.y_rotations, expected.y_rotations_)
    assert np.allclose(state.coef, expected.coef_)
    assert np.allclose(state.intercept, expected.intercept_)
    assert np.allclose(state.x_mean, expected._x_mean)
    assert np.allclose(state.y_mean, expected._y_mean)
    assert np.allclose(state.x_std, expected._x_std)
    assert np.allclose(state.y_std, expected._y_std)
    assert state.n_iter == tuple(expected.n_iter_)
    assert state.n_features_in == expected.n_features_in_


def test_pls_fit_imports() -> None:
    from sciona.atoms.ml.sklearn.cross_decomposition import PLSState, cca_fit, pls_canonical_fit, pls_regression_fit

    assert PLSState is not None
    assert callable(cca_fit)
    assert callable(pls_canonical_fit)
    assert callable(pls_regression_fit)


def test_pls_regression_fit_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.cross_decomposition import pls_regression_fit

    X, y = _xy()
    state = pls_regression_fit(X, y, n_components=2, scale=True)
    expected = SklearnPLSRegression(n_components=2, scale=True).fit(X, y)

    _assert_common_state(state, expected)
    assert state.deflation_mode == "regression"
    assert state.mode == "A"
    assert state.algorithm == "nipals"


def test_pls_canonical_fit_matches_sklearn_nipals_and_svd() -> None:
    from sciona.atoms.ml.sklearn.cross_decomposition import pls_canonical_fit

    X, y = _xy()
    state = pls_canonical_fit(X, y, n_components=2, algorithm="nipals")
    expected = SklearnPLSCanonical(n_components=2, algorithm="nipals").fit(X, y)
    _assert_common_state(state, expected)

    svd_state = pls_canonical_fit(X, y, n_components=2, algorithm="svd", scale=False)
    svd_expected = SklearnPLSCanonical(n_components=2, algorithm="svd", scale=False).fit(X, y)
    _assert_common_state(svd_state, svd_expected)
    assert svd_state.n_iter == ()


def test_cca_fit_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.cross_decomposition import cca_fit

    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 4.0],
            [4.0, 8.0],
            [5.0, 9.0],
            [6.0, 11.0],
        ],
        dtype=np.float64,
    )
    y = np.array([[0.0, 1.0], [1.0, 1.5], [2.0, 2.5], [3.0, 4.0], [5.0, 6.0], [8.0, 9.0]], dtype=np.float64)

    state = cca_fit(X, y, n_components=1, scale=True)
    expected = SklearnCCA(n_components=1, scale=True).fit(X, y)

    _assert_common_state(state, expected)
    assert state.deflation_mode == "canonical"
    assert state.mode == "B"


def test_pls_fit_rejects_too_many_components() -> None:
    from sciona.atoms.ml.sklearn.cross_decomposition import cca_fit, pls_regression_fit

    X = np.ones((4, 3), dtype=np.float64)
    y = np.ones((4, 1), dtype=np.float64)

    with pytest.raises(Exception):
        cca_fit(X, y, n_components=2)

    with pytest.raises(Exception):
        pls_regression_fit(X, y, n_components=5)
