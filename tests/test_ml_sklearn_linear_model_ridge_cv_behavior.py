from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV as SklearnRidgeCV


def _data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 3.0],
            [4.0, 2.0],
            [5.0, 4.0],
        ],
        dtype=np.float64,
    )
    y = X @ np.array([1.5, -0.75], dtype=np.float64) + 2.0
    return X, y


def _naive_loo_scores(X: np.ndarray, y: np.ndarray, alphas: np.ndarray, *, fit_intercept: bool = True) -> np.ndarray:
    y_2d = y.reshape(-1, 1) if y.ndim == 1 else y
    scores = np.empty(alphas.shape[0], dtype=np.float64)
    indices = np.arange(X.shape[0])
    for alpha_index, alpha in enumerate(alphas):
        predictions = np.empty_like(y_2d, dtype=np.float64)
        for held_out in range(X.shape[0]):
            train_mask = indices != held_out
            train_y = y[train_mask]
            estimator = Ridge(alpha=float(alpha), fit_intercept=fit_intercept, solver="cholesky").fit(X[train_mask], train_y)
            predictions[held_out] = np.asarray(estimator.predict(X[held_out : held_out + 1])).reshape(1, -1)[0]
        scores[alpha_index] = -float(np.mean((y_2d - predictions) ** 2))
    return scores


def test_ridge_cv_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model import RidgeCVState, ridge_cv_fit, ridge_cv_predict, ridge_cv_scores

    assert RidgeCVState is not None
    assert callable(ridge_cv_scores)
    assert callable(ridge_cv_fit)
    assert callable(ridge_cv_predict)


def test_ridge_cv_scores_match_leave_one_out_scores() -> None:
    from sciona.atoms.ml.sklearn.linear_model import ridge_cv_scores

    X, y = _data()
    alphas = np.array([0.1, 0.5, 1.0, 5.0], dtype=np.float64)
    result = ridge_cv_scores(X, y, alphas)
    expected = _naive_loo_scores(X, y, alphas)

    assert np.allclose(result, expected)


def test_ridge_cv_fit_and_predict_match_sklearn_single_output() -> None:
    from sciona.atoms.ml.sklearn.linear_model import ridge_cv_fit, ridge_cv_predict

    X, y = _data()
    alphas = np.array([0.1, 0.5, 1.0, 5.0], dtype=np.float64)
    state = ridge_cv_fit(X, y, alphas=alphas)
    expected = SklearnRidgeCV(alphas=alphas, cv=None, scoring=None).fit(X, y)

    assert np.allclose(state.alpha[0], expected.alpha_)
    assert np.allclose(state.best_score[0], expected.best_score_)
    assert np.allclose(state.coef, expected.coef_)
    assert np.allclose(state.intercept[0], expected.intercept_)
    assert np.allclose(ridge_cv_predict(X[:3], state), expected.predict(X[:3]))


def test_ridge_cv_fit_and_predict_match_sklearn_multioutput_without_intercept() -> None:
    from sciona.atoms.ml.sklearn.linear_model import ridge_cv_fit, ridge_cv_predict

    X, y_1d = _data()
    y = np.column_stack([y_1d, 0.25 * y_1d + 1.5])
    alphas = np.array([0.1, 0.75, 2.0], dtype=np.float64)
    state = ridge_cv_fit(X, y, alphas=alphas, fit_intercept=False, gcv_mode="svd")
    expected = SklearnRidgeCV(alphas=alphas, fit_intercept=False, cv=None, scoring=None, gcv_mode="svd").fit(X, y)

    assert np.allclose(state.alpha[0], expected.alpha_)
    assert np.allclose(state.best_score[0], expected.best_score_)
    assert np.allclose(state.coef, expected.coef_)
    assert np.allclose(state.intercept, expected.intercept_)
    assert np.allclose(ridge_cv_predict(X[:2], state), expected.predict(X[:2]))


def test_ridge_cv_rejects_unsupported_modes() -> None:
    from sciona.atoms.ml.sklearn.linear_model import ridge_cv_fit, ridge_cv_predict, ridge_cv_scores

    X, y = _data()
    state = ridge_cv_fit(X, y)
    with pytest.raises(Exception):
        ridge_cv_scores(X[:1], y[:1])
    with pytest.raises(Exception):
        ridge_cv_scores(X, y, alphas=(0.0, 1.0))
    with pytest.raises(Exception):
        ridge_cv_fit(X, y, scoring="r2")
    with pytest.raises(Exception):
        ridge_cv_fit(X, y, cv=3)
    with pytest.raises(Exception):
        ridge_cv_fit(X, y, sample_weight=np.ones(X.shape[0], dtype=np.float64))
    with pytest.raises(Exception):
        ridge_cv_predict(np.ones((2, 3), dtype=np.float64), state)
