from __future__ import annotations

import numpy as np
import pytest
from sklearn.mixture import GaussianMixture as SklearnGaussianMixture
from sklearn.mixture import BayesianGaussianMixture as SklearnBayesianGaussianMixture


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
        BayesianGaussianMixtureDiagState,
        GaussianMixtureDiagState,
        bayesian_gaussian_mixture_diag_fit,
        bayesian_gaussian_mixture_diag_predict,
        bayesian_gaussian_mixture_diag_predict_proba,
        bayesian_gaussian_mixture_diag_score,
        bayesian_gaussian_mixture_diag_score_samples,
        gaussian_mixture_diag_aic,
        gaussian_mixture_diag_bic,
        gaussian_mixture_diag_fit,
        gaussian_mixture_diag_predict,
        gaussian_mixture_diag_predict_proba,
        gaussian_mixture_diag_score,
        gaussian_mixture_diag_score_samples,
    )

    assert BayesianGaussianMixtureDiagState is not None
    assert GaussianMixtureDiagState is not None
    assert callable(bayesian_gaussian_mixture_diag_fit)
    assert callable(bayesian_gaussian_mixture_diag_score_samples)
    assert callable(bayesian_gaussian_mixture_diag_score)
    assert callable(bayesian_gaussian_mixture_diag_predict_proba)
    assert callable(bayesian_gaussian_mixture_diag_predict)
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


def test_bayesian_gaussian_mixture_diag_fit_and_scores_match_sklearn_private_updates() -> None:
    from sciona.atoms.ml.sklearn.mixture import (
        bayesian_gaussian_mixture_diag_fit,
        bayesian_gaussian_mixture_diag_predict,
        bayesian_gaussian_mixture_diag_predict_proba,
        bayesian_gaussian_mixture_diag_score,
        bayesian_gaussian_mixture_diag_score_samples,
    )

    X, _, _, _ = _fixture()
    responsibilities = np.array(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    state = bayesian_gaussian_mixture_diag_fit(X, responsibilities, max_iter=10, tol=1e-9)

    expected = SklearnBayesianGaussianMixture(
        n_components=2,
        covariance_type="diag",
        max_iter=10,
        tol=1e-9,
        reg_covar=1e-6,
        n_init=1,
    )
    expected._check_parameters(X)
    expected._initialize(X, responsibilities)
    lower_bound = -np.inf
    lower_bounds = []
    converged = False
    for n_iter in range(1, expected.max_iter + 1):
        previous_lower_bound = lower_bound
        log_prob_norm, log_resp = expected._e_step(X)
        expected._m_step(X, log_resp)
        lower_bound = expected._compute_lower_bound(log_resp, log_prob_norm)
        lower_bounds.append(lower_bound)
        if abs(lower_bound - previous_lower_bound) < expected.tol:
            converged = True
            break
    expected._set_parameters(expected._get_parameters())
    expected.n_iter_ = n_iter
    expected.converged_ = converged
    expected.lower_bound_ = lower_bound
    expected.lower_bounds_ = lower_bounds

    assert np.allclose(state.weights, expected.weights_)
    assert np.allclose(state.means, expected.means_)
    assert np.allclose(state.covariances, expected.covariances_)
    assert np.allclose(state.mean_precision, expected.mean_precision_)
    assert np.allclose(state.degrees_of_freedom, expected.degrees_of_freedom_)
    assert np.allclose(state.weight_concentration, np.asarray(expected.weight_concentration_))
    assert state.n_iter == expected.n_iter_
    assert state.converged == expected.converged_
    assert np.isclose(state.lower_bound, expected.lower_bound_)
    assert np.allclose(bayesian_gaussian_mixture_diag_score_samples(X, state), expected.score_samples(X))
    assert np.isclose(bayesian_gaussian_mixture_diag_score(X, state), expected.score(X))
    assert np.allclose(bayesian_gaussian_mixture_diag_predict_proba(X, state), expected.predict_proba(X))
    assert np.array_equal(bayesian_gaussian_mixture_diag_predict(X, state), expected.predict(X))


def test_bayesian_gaussian_mixture_diag_rejects_invalid_scope() -> None:
    from sciona.atoms.ml.sklearn.mixture import (
        bayesian_gaussian_mixture_diag_fit,
        bayesian_gaussian_mixture_diag_predict,
    )

    X, _, _, _ = _fixture()
    responsibilities = np.array(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    with pytest.raises(Exception):
        bayesian_gaussian_mixture_diag_fit(X, responsibilities[:, :1] * 0.0)
    with pytest.raises(Exception):
        bayesian_gaussian_mixture_diag_fit(X, responsibilities, max_iter=0)
    with pytest.raises(Exception):
        bayesian_gaussian_mixture_diag_fit(X, responsibilities, weight_concentration_prior_type="bad")
    with pytest.raises(Exception):
        bayesian_gaussian_mixture_diag_fit(X, responsibilities, degrees_of_freedom_prior=0.5)
    state = bayesian_gaussian_mixture_diag_fit(X, responsibilities, max_iter=2)
    with pytest.raises(Exception):
        bayesian_gaussian_mixture_diag_predict(X[:, :1], state)
