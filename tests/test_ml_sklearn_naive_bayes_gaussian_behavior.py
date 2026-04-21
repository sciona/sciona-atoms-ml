from __future__ import annotations

import numpy as np
import pytest
from sklearn.naive_bayes import GaussianNB


def _training_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [-2.0, -1.0, 0.0],
            [-1.5, -1.2, 0.3],
            [-0.8, -0.7, 0.5],
            [0.6, 0.8, 1.0],
            [1.2, 0.9, 1.4],
            [1.8, 1.5, 1.9],
        ],
        dtype=np.float64,
    )
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    return X, y


def test_gaussian_nb_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.naive_bayes import (
        GaussianNBState,
        gaussian_nb_fit,
        gaussian_nb_joint_log_likelihood,
        gaussian_nb_predict,
        gaussian_nb_predict_log_proba,
        gaussian_nb_predict_proba,
        gaussian_nb_update_mean_variance,
    )

    assert GaussianNBState is not None
    assert callable(gaussian_nb_fit)
    assert callable(gaussian_nb_joint_log_likelihood)
    assert callable(gaussian_nb_predict)
    assert callable(gaussian_nb_predict_log_proba)
    assert callable(gaussian_nb_predict_proba)
    assert callable(gaussian_nb_update_mean_variance)


def test_gaussian_nb_update_mean_variance_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.naive_bayes import gaussian_nb_update_mean_variance

    X, _ = _training_data()
    old_mu = np.array([0.2, 0.1, 0.4], dtype=np.float64)
    old_var = np.array([0.4, 0.5, 0.8], dtype=np.float64)
    weights = np.array([1.0, 2.0, 1.5, 0.75, 1.25, 2.5], dtype=np.float64)

    actual_mu, actual_var = gaussian_nb_update_mean_variance(3.5, old_mu, old_var, X, weights)
    expected_mu, expected_var = GaussianNB._update_mean_variance(3.5, old_mu, old_var, X, weights)

    assert np.allclose(actual_mu, expected_mu)
    assert np.allclose(actual_var, expected_var)


def test_gaussian_nb_fit_state_matches_sklearn_unweighted() -> None:
    from sciona.atoms.ml.sklearn.naive_bayes import gaussian_nb_fit

    X, y = _training_data()
    expected = GaussianNB().fit(X, y)
    state = gaussian_nb_fit(X, y)

    assert np.array_equal(state.classes, expected.classes_)
    assert np.allclose(state.class_count, expected.class_count_)
    assert np.allclose(state.class_prior, expected.class_prior_)
    assert np.allclose(state.theta, expected.theta_)
    assert np.allclose(state.var, expected.var_)
    assert np.isclose(state.epsilon, expected.epsilon_)


def test_gaussian_nb_fit_state_matches_sklearn_weighted_with_priors() -> None:
    from sciona.atoms.ml.sklearn.naive_bayes import gaussian_nb_fit

    X, y = _training_data()
    priors = np.array([0.35, 0.65], dtype=np.float64)
    weights = np.array([1.0, 1.5, 2.0, 1.25, 2.25, 1.75], dtype=np.float64)
    expected = GaussianNB(priors=priors).fit(X, y, sample_weight=weights)
    state = gaussian_nb_fit(X, y, priors=priors, sample_weight=weights)

    assert np.array_equal(state.classes, expected.classes_)
    assert np.allclose(state.class_count, expected.class_count_)
    assert np.allclose(state.class_prior, expected.class_prior_)
    assert np.allclose(state.theta, expected.theta_)
    assert np.allclose(state.var, expected.var_)


def test_gaussian_nb_predictions_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.naive_bayes import (
        gaussian_nb_fit,
        gaussian_nb_joint_log_likelihood,
        gaussian_nb_predict,
        gaussian_nb_predict_log_proba,
        gaussian_nb_predict_proba,
    )

    X, y = _training_data()
    query = np.array([[-1.2, -1.1, 0.2], [1.0, 1.2, 1.3], [0.0, 0.1, 0.8]], dtype=np.float64)
    expected = GaussianNB().fit(X, y)
    state = gaussian_nb_fit(X, y)

    assert np.allclose(gaussian_nb_joint_log_likelihood(query, state), expected._joint_log_likelihood(query))
    assert np.allclose(gaussian_nb_predict_log_proba(query, state), expected.predict_log_proba(query))
    assert np.allclose(gaussian_nb_predict_proba(query, state), expected.predict_proba(query))
    assert np.array_equal(gaussian_nb_predict(query, state), expected.predict(query))


def test_gaussian_nb_rejects_unsupported_inputs() -> None:
    from sciona.atoms.ml.sklearn.naive_bayes import gaussian_nb_fit, gaussian_nb_predict

    X, y = _training_data()
    state = gaussian_nb_fit(X, y)

    with pytest.raises(Exception):
        gaussian_nb_fit(X[:, :1] * 0.0, y)
    with pytest.raises(Exception):
        gaussian_nb_fit(X, np.zeros(X.shape[0], dtype=np.int64))
    with pytest.raises(Exception):
        gaussian_nb_fit(X, y, priors=np.array([0.0, 1.0], dtype=np.float64))
    with pytest.raises(Exception):
        gaussian_nb_fit(X, y, sample_weight=np.array([1.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64))
    with pytest.raises(Exception):
        gaussian_nb_predict(np.ones((2, 2), dtype=np.float64), state)
