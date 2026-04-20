from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import OrthogonalMatchingPursuit as SklearnOrthogonalMatchingPursuit
from sklearn.linear_model import orthogonal_mp as sklearn_orthogonal_mp
from sklearn.linear_model import orthogonal_mp_gram as sklearn_orthogonal_mp_gram


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    y = X @ np.array([2.0, 0.0, -1.0, 0.0], dtype=np.float64) + 0.5
    y_multi = np.column_stack([y, X @ np.array([0.0, 3.0, 0.0, -2.0], dtype=np.float64) - 1.0])
    return X, y, y_multi


def test_omp_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model import (
        OrthogonalMatchingPursuitState,
        orthogonal_matching_pursuit_fit,
        orthogonal_matching_pursuit_predict,
        orthogonal_mp,
        orthogonal_mp_gram,
    )

    assert OrthogonalMatchingPursuitState is not None
    assert callable(orthogonal_mp)
    assert callable(orthogonal_mp_gram)
    assert callable(orthogonal_matching_pursuit_fit)
    assert callable(orthogonal_matching_pursuit_predict)


def test_orthogonal_mp_matches_sklearn_single_output() -> None:
    from sciona.atoms.ml.sklearn.linear_model import orthogonal_mp

    X, y, _ = _data()
    result = orthogonal_mp(X, y, n_nonzero_coefs=2, precompute=False)
    expected = sklearn_orthogonal_mp(X, y, n_nonzero_coefs=2, precompute=False)

    assert np.allclose(result, expected)


def test_orthogonal_mp_precomputed_matches_sklearn_multioutput() -> None:
    from sciona.atoms.ml.sklearn.linear_model import orthogonal_mp

    X, _, y = _data()
    result = orthogonal_mp(X, y, n_nonzero_coefs=2, precompute=True)
    expected = sklearn_orthogonal_mp(X, y, n_nonzero_coefs=2, precompute=True)

    assert np.allclose(result, expected)


def test_orthogonal_mp_gram_matches_sklearn_multioutput() -> None:
    from sciona.atoms.ml.sklearn.linear_model import orthogonal_mp_gram

    X, _, y = _data()
    gram = X.T @ X
    xy = X.T @ y
    result = orthogonal_mp_gram(gram, xy, n_nonzero_coefs=2)
    expected = sklearn_orthogonal_mp_gram(gram, xy, n_nonzero_coefs=2)

    assert np.allclose(result, expected)


def test_orthogonal_matching_pursuit_fit_predict_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.linear_model import orthogonal_matching_pursuit_fit, orthogonal_matching_pursuit_predict

    X, y, _ = _data()
    state = orthogonal_matching_pursuit_fit(X, y, n_nonzero_coefs=2)
    expected = SklearnOrthogonalMatchingPursuit(n_nonzero_coefs=2).fit(X, y)

    assert np.allclose(state.coef, expected.coef_)
    assert np.allclose(state.intercept[0], expected.intercept_)
    assert state.n_features_in == expected.n_features_in_
    assert np.array_equal(state.n_iter, np.atleast_1d(expected.n_iter_))
    assert np.allclose(orthogonal_matching_pursuit_predict(X[:3], state), expected.predict(X[:3]))


def test_orthogonal_matching_pursuit_fit_predict_matches_sklearn_multioutput_without_intercept() -> None:
    from sciona.atoms.ml.sklearn.linear_model import orthogonal_matching_pursuit_fit, orthogonal_matching_pursuit_predict

    X, _, y = _data()
    state = orthogonal_matching_pursuit_fit(X, y, n_nonzero_coefs=2, fit_intercept=False, precompute=False)
    expected = SklearnOrthogonalMatchingPursuit(n_nonzero_coefs=2, fit_intercept=False, precompute=False).fit(X, y)

    assert np.allclose(state.coef, expected.coef_)
    assert np.allclose(state.intercept, expected.intercept_)
    assert np.array_equal(state.n_iter, np.asarray(expected.n_iter_))
    assert np.allclose(orthogonal_matching_pursuit_predict(X[:2], state), expected.predict(X[:2]))


def test_omp_rejects_unsupported_or_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.linear_model import orthogonal_matching_pursuit_fit, orthogonal_matching_pursuit_predict, orthogonal_mp, orthogonal_mp_gram

    X, y, _ = _data()
    state = orthogonal_matching_pursuit_fit(X, y, n_nonzero_coefs=2)
    with pytest.raises(Exception):
        orthogonal_mp(X, y[:-1], n_nonzero_coefs=2)
    with pytest.raises(Exception):
        orthogonal_mp(X, y, n_nonzero_coefs=5)
    with pytest.raises(Exception):
        orthogonal_mp(X, y, n_nonzero_coefs=2, return_path=True)
    with pytest.raises(Exception):
        orthogonal_mp_gram(X.T @ X, X.T @ y, n_nonzero_coefs=2, tol=0.1)
    with pytest.raises(Exception):
        orthogonal_matching_pursuit_fit(X, y, tol=-1.0)
    with pytest.raises(Exception):
        orthogonal_matching_pursuit_predict(np.ones((2, 3), dtype=np.float64), state)
