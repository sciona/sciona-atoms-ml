from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import OrthogonalMatchingPursuitCV as SklearnOrthogonalMatchingPursuitCV
from sklearn.linear_model._omp import _omp_path_residues as sklearn_omp_path_residues


def _data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(15, 6)).astype(np.float64)
    y = X @ np.array([2.0, 0.0, -1.0, 0.0, 0.5, 0.0], dtype=np.float64) + 0.25
    return X, y


def test_omp_cv_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model import (
        OrthogonalMatchingPursuitCVState,
        omp_path_residues,
        orthogonal_matching_pursuit_cv_fit,
        orthogonal_matching_pursuit_cv_predict,
    )

    assert OrthogonalMatchingPursuitCVState is not None
    assert callable(omp_path_residues)
    assert callable(orthogonal_matching_pursuit_cv_fit)
    assert callable(orthogonal_matching_pursuit_cv_predict)


def test_omp_path_residues_match_sklearn_private_helper() -> None:
    from sciona.atoms.ml.sklearn.linear_model import omp_path_residues

    X, y = _data()
    result = omp_path_residues(X[:10], y[:10], X[10:], y[10:], max_iter=4)
    expected = sklearn_omp_path_residues(X[:10], y[:10], X[10:], y[10:], max_iter=4)

    assert np.allclose(result, expected)


def test_omp_cv_fit_predict_matches_sklearn_default_cv() -> None:
    from sciona.atoms.ml.sklearn.linear_model import orthogonal_matching_pursuit_cv_fit, orthogonal_matching_pursuit_cv_predict

    X, y = _data()
    state = orthogonal_matching_pursuit_cv_fit(X, y)
    expected = SklearnOrthogonalMatchingPursuitCV().fit(X, y)

    assert state.n_nonzero_coefs == expected.n_nonzero_coefs_
    assert np.allclose(state.coef, expected.coef_)
    assert np.allclose(state.intercept[0], expected.intercept_)
    assert np.array_equal(state.n_iter, np.atleast_1d(expected.n_iter_))
    assert np.allclose(orthogonal_matching_pursuit_cv_predict(X[:4], state), expected.predict(X[:4]))


def test_omp_cv_fit_predict_matches_sklearn_explicit_cv_without_intercept() -> None:
    from sciona.atoms.ml.sklearn.linear_model import orthogonal_matching_pursuit_cv_fit, orthogonal_matching_pursuit_cv_predict

    X, y = _data()
    state = orthogonal_matching_pursuit_cv_fit(X, y, fit_intercept=False, cv=5, max_iter=4)
    expected = SklearnOrthogonalMatchingPursuitCV(fit_intercept=False, cv=5, max_iter=4).fit(X, y)

    assert state.n_nonzero_coefs == expected.n_nonzero_coefs_
    assert np.allclose(state.coef, expected.coef_)
    assert np.allclose(state.intercept[0], expected.intercept_)
    assert np.array_equal(state.n_iter, np.atleast_1d(expected.n_iter_))
    assert np.allclose(orthogonal_matching_pursuit_cv_predict(X[:4], state), expected.predict(X[:4]))


def test_omp_cv_rejects_invalid_or_unsupported_inputs() -> None:
    from sciona.atoms.ml.sklearn.linear_model import orthogonal_matching_pursuit_cv_fit, orthogonal_matching_pursuit_cv_predict

    X, y = _data()
    state = orthogonal_matching_pursuit_cv_fit(X, y)
    with pytest.raises(Exception):
        orthogonal_matching_pursuit_cv_fit(X, y[:-1])
    with pytest.raises(Exception):
        orthogonal_matching_pursuit_cv_fit(X, y, cv=1)
    with pytest.raises(Exception):
        orthogonal_matching_pursuit_cv_fit(X, y, max_iter=7)
    with pytest.raises(Exception):
        orthogonal_matching_pursuit_cv_fit(X, y, n_jobs=2)
    with pytest.raises(Exception):
        orthogonal_matching_pursuit_cv_predict(np.ones((2, 3), dtype=np.float64), state)
