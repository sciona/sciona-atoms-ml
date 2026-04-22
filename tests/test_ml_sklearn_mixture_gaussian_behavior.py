from __future__ import annotations

import numpy as np
import pytest
from sklearn.mixture import GaussianMixture as SklearnGaussianMixture


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [-0.1, 0.0],
            [5.0, 5.0],
            [5.2, 5.1],
            [4.8, 4.9],
        ],
        dtype=np.float64,
    )
    weights = np.array([0.5, 0.5], dtype=np.float64)
    means = np.array([[0.0, 0.0], [5.0, 5.0]], dtype=np.float64)
    variances = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    return X, weights, means, variances


def test_gaussian_mixture_diag_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.mixture import (
        GaussianMixtureDiagState,
        gaussian_mixture_diag_aic,
        gaussian_mixture_diag_bic,
        gaussian_mixture_diag_fit,
        gaussian_mixture_diag_predict,
        gaussian_mixture_diag_predict_proba,
        gaussian_mixture_diag_score,
        gaussian_mixture_diag_score_samples,
    )

    assert GaussianMixtureDiagState is not None
    assert callable(gaussian_mixture_diag_fit)
    assert callable(gaussian_mixture_diag_score_samples)
    assert callable(gaussian_mixture_diag_score)
    assert callable(gaussian_mixture_diag_predict_proba)
    assert callable(gaussian_mixture_diag_predict)
    assert callable(gaussian_mixture_diag_bic)
    assert callable(gaussian_mixture_diag_aic)


def test_gaussian_mixture_diag_fit_and_scores_match_sklearn_explicit_initialization() -> None:
    from sciona.atoms.ml.sklearn.mixture import (
        gaussian_mixture_diag_aic,
        gaussian_mixture_diag_bic,
        gaussian_mixture_diag_fit,
        gaussian_mixture_diag_predict,
        gaussian_mixture_diag_predict_proba,
        gaussian_mixture_diag_score,
        gaussian_mixture_diag_score_samples,
    )

    X, weights, means, variances = _fixture()
    state = gaussian_mixture_diag_fit(
        X,
        weights,
        means,
        variances,
        max_iter=20,
        tol=1e-9,
    )
    expected = SklearnGaussianMixture(
        n_components=2,
        covariance_type="diag",
        weights_init=weights,
        means_init=means,
        precisions_init=1.0 / variances,
        max_iter=20,
        tol=1e-9,
        reg_covar=1e-6,
        n_init=1,
    ).fit(X)

    assert np.allclose(state.weights, expected.weights_)
    assert np.allclose(state.means, expected.means_)
    assert np.allclose(state.covariances, expected.covariances_)
    assert state.n_iter == expected.n_iter_
    assert state.converged == expected.converged_
    assert np.allclose(gaussian_mixture_diag_score_samples(X, state), expected.score_samples(X))
    assert np.isclose(gaussian_mixture_diag_score(X, state), expected.score(X))
    assert np.allclose(gaussian_mixture_diag_predict_proba(X, state), expected.predict_proba(X))
    assert np.array_equal(gaussian_mixture_diag_predict(X, state), expected.predict(X))
    assert np.isclose(gaussian_mixture_diag_bic(X, state), expected.bic(X))
    assert np.isclose(gaussian_mixture_diag_aic(X, state), expected.aic(X))


def test_gaussian_mixture_diag_rejects_invalid_scope() -> None:
    from sciona.atoms.ml.sklearn.mixture import gaussian_mixture_diag_fit, gaussian_mixture_diag_predict

    X, weights, means, variances = _fixture()
    with pytest.raises(Exception):
        gaussian_mixture_diag_fit(X, np.array([0.2, 0.2]), means, variances)
    with pytest.raises(Exception):
        gaussian_mixture_diag_fit(X, weights, means[:, :1], variances)
    with pytest.raises(Exception):
        gaussian_mixture_diag_fit(X, weights, means, -variances)
    state = gaussian_mixture_diag_fit(X, weights, means, variances, max_iter=1)
    with pytest.raises(Exception):
        gaussian_mixture_diag_predict(X[:, :1], state)
