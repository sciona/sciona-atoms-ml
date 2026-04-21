from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Lars as SklearnLars
from sklearn.linear_model import lars_path as sklearn_lars_path
from sklearn.linear_model import lars_path_gram as sklearn_lars_path_gram


def _data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 0.5],
            [3.0, 4.0, 1.5],
            [4.0, 2.0, 2.5],
            [5.0, 3.5, 0.25],
        ],
        dtype=np.float64,
    )
    y = X @ np.array([1.2, -0.7, 2.1], dtype=np.float64) + 0.5
    return X, y


def test_lars_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model import (
        LarsPathState,
        LarsState,
        lars_fit,
        lars_path,
        lars_path_gram,
        lars_predict,
    )

    assert LarsPathState is not None
    assert LarsState is not None
    assert callable(lars_path)
    assert callable(lars_path_gram)
    assert callable(lars_fit)
    assert callable(lars_predict)


def test_lars_path_matches_sklearn_lar_method() -> None:
    from sciona.atoms.ml.sklearn.linear_model import lars_path

    X, y = _data()
    state = lars_path(X, y, max_iter=3)
    alphas, active, coefs = sklearn_lars_path(X, y, method="lar", max_iter=3)

    assert np.allclose(state.alphas, alphas)
    assert np.array_equal(state.active, np.asarray(active, dtype=np.int64))
    assert np.allclose(state.coefs, coefs)
    assert state.n_iter == len(active)


def test_lars_path_gram_matches_sklearn_lar_method() -> None:
    from sciona.atoms.ml.sklearn.linear_model import lars_path_gram

    X, y = _data()
    Xy = X.T @ y
    Gram = X.T @ X
    state = lars_path_gram(Xy, Gram, n_samples=X.shape[0], max_iter=3)
    alphas, active, coefs = sklearn_lars_path_gram(Xy, Gram, n_samples=X.shape[0], method="lar", max_iter=3)

    assert np.allclose(state.alphas, alphas)
    assert np.array_equal(state.active, np.asarray(active, dtype=np.int64))
    assert np.allclose(state.coefs, coefs)


def test_lars_fit_predict_matches_sklearn_default_scope() -> None:
    from sciona.atoms.ml.sklearn.linear_model import lars_fit, lars_predict

    X, y = _data()
    state = lars_fit(X, y, n_nonzero_coefs=3)
    expected = SklearnLars(n_nonzero_coefs=3).fit(X, y)

    assert np.allclose(state.coef, expected.coef_)
    assert np.isclose(state.intercept, expected.intercept_)
    assert np.allclose(state.alphas, expected.alphas_)
    assert np.array_equal(state.active, np.asarray(expected.active_, dtype=np.int64))
    assert np.allclose(state.coef_path, expected.coef_path_)
    assert state.n_iter == expected.n_iter_
    assert np.allclose(lars_predict(X, state), expected.predict(X))


def test_lars_fit_without_intercept_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.linear_model import lars_fit, lars_predict

    X, y = _data()
    state = lars_fit(X, y, fit_intercept=False, n_nonzero_coefs=2)
    expected = SklearnLars(fit_intercept=False, n_nonzero_coefs=2).fit(X, y)

    assert np.allclose(state.coef, expected.coef_)
    assert np.isclose(state.intercept, expected.intercept_)
    assert np.allclose(state.alphas, expected.alphas_)
    assert np.allclose(lars_predict(X, state), expected.predict(X))


def test_lars_rejects_out_of_scope_inputs() -> None:
    from sciona.atoms.ml.sklearn.linear_model import lars_fit, lars_path, lars_predict

    X, y = _data()
    with pytest.raises(Exception):
        lars_path(X, y, method="lasso")
    with pytest.raises(Exception):
        lars_path(X, y, positive=True)
    with pytest.raises(Exception):
        lars_path(X, y, return_path=False)
    with pytest.raises(Exception):
        lars_fit(X, y, jitter=1e-4)
    with pytest.raises(Exception):
        lars_fit(X, y, fit_path=False)

    state = lars_fit(X, y)
    with pytest.raises(Exception):
        lars_predict(np.ones((2, 2), dtype=np.float64), state)
