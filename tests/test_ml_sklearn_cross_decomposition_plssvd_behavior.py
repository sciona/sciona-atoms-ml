from __future__ import annotations

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSSVD as SklearnPLSSVD


def test_plssvd_fit_import() -> None:
    from sciona.atoms.ml.sklearn.cross_decomposition import PLSSVDState, plssvd_fit

    assert PLSSVDState is not None
    assert callable(plssvd_fit)


def test_plssvd_fit_matches_sklearn_scaled() -> None:
    from sciona.atoms.ml.sklearn.cross_decomposition import plssvd_fit

    X = np.array(
        [
            [0.0, 1.0, 3.0],
            [1.0, 2.0, 4.0],
            [2.0, 4.0, 7.0],
            [4.0, 8.0, 9.0],
            [5.0, 9.0, 12.0],
        ],
        dtype=np.float64,
    )
    y = np.array([[0.0, 1.0], [1.0, 1.5], [2.0, 2.5], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)

    state = plssvd_fit(X, y, n_components=2, scale=True)
    expected = SklearnPLSSVD(n_components=2, scale=True).fit(X, y)

    assert np.allclose(state.x_weights, expected.x_weights_)
    assert np.allclose(state.y_weights, expected.y_weights_)
    assert np.allclose(state.x_mean, expected._x_mean)
    assert np.allclose(state.y_mean, expected._y_mean)
    assert np.allclose(state.x_std, expected._x_std)
    assert np.allclose(state.y_std, expected._y_std)
    assert state.n_features_in == expected.n_features_in_


def test_plssvd_fit_matches_sklearn_unscaled_1d_target() -> None:
    from sciona.atoms.ml.sklearn.cross_decomposition import plssvd_fit

    X = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0], [4.0, 8.0]], dtype=np.float64)
    y = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)

    state = plssvd_fit(X, y, n_components=1, scale=False)
    expected = SklearnPLSSVD(n_components=1, scale=False).fit(X, y)

    assert np.allclose(state.x_weights, expected.x_weights_)
    assert np.allclose(state.y_weights, expected.y_weights_)
    assert state.n_targets == 1
    assert np.allclose(state.x_std, np.ones(2))
    assert np.allclose(state.y_std, np.ones(1))


def test_plssvd_fit_rejects_too_many_components() -> None:
    from sciona.atoms.ml.sklearn.cross_decomposition import plssvd_fit

    X = np.ones((4, 3), dtype=np.float64)
    y = np.ones((4, 1), dtype=np.float64)

    with pytest.raises(Exception):
        plssvd_fit(X, y, n_components=2)
