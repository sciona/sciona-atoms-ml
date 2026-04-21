from __future__ import annotations

import numpy as np
import pytest
from sklearn.naive_bayes import MultinomialNB


def _training_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [2.0, 1.0, 0.0, 0.0],
            [3.0, 2.0, 0.0, 1.0],
            [1.0, 2.0, 0.0, 0.0],
            [0.0, 1.0, 3.0, 2.0],
            [0.0, 0.0, 4.0, 3.0],
            [1.0, 0.0, 2.0, 4.0],
        ],
        dtype=np.float64,
    )
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    return X, y


def test_multinomial_nb_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.naive_bayes import (
        MultinomialNBState,
        multinomial_nb_class_log_prior,
        multinomial_nb_count,
        multinomial_nb_feature_log_prob,
        multinomial_nb_fit,
        multinomial_nb_joint_log_likelihood,
        multinomial_nb_predict,
        multinomial_nb_predict_log_proba,
        multinomial_nb_predict_proba,
    )

    assert MultinomialNBState is not None
    assert callable(multinomial_nb_class_log_prior)
    assert callable(multinomial_nb_count)
    assert callable(multinomial_nb_feature_log_prob)
    assert callable(multinomial_nb_fit)
    assert callable(multinomial_nb_joint_log_likelihood)
    assert callable(multinomial_nb_predict)
    assert callable(multinomial_nb_predict_log_proba)
    assert callable(multinomial_nb_predict_proba)


def test_multinomial_nb_count_and_feature_log_prob_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.naive_bayes import (
        multinomial_nb_count,
        multinomial_nb_feature_log_prob,
    )

    X, y = _training_data()
    weights = np.array([1.0, 2.0, 1.5, 1.25, 1.75, 2.5], dtype=np.float64)
    expected = MultinomialNB(alpha=0.75).fit(X, y, sample_weight=weights)
    classes, class_count, feature_count = multinomial_nb_count(X, y, weights)
    feature_log_prob = multinomial_nb_feature_log_prob(feature_count, alpha=0.75)

    assert np.array_equal(classes, expected.classes_)
    assert np.allclose(class_count, expected.class_count_)
    assert np.allclose(feature_count, expected.feature_count_)
    assert np.allclose(feature_log_prob, expected.feature_log_prob_)


def test_multinomial_nb_class_log_prior_matches_sklearn_modes() -> None:
    from sciona.atoms.ml.sklearn.naive_bayes import multinomial_nb_class_log_prior

    counts = np.array([2.5, 7.5], dtype=np.float64)
    priors = np.array([0.3, 0.7], dtype=np.float64)

    assert np.allclose(multinomial_nb_class_log_prior(counts), np.log(counts) - np.log(np.sum(counts)))
    assert np.allclose(multinomial_nb_class_log_prior(counts, fit_prior=False), np.array([-np.log(2.0), -np.log(2.0)]))
    assert np.allclose(multinomial_nb_class_log_prior(counts, class_prior=priors), np.log(priors))


def test_multinomial_nb_fit_state_matches_sklearn_unweighted() -> None:
    from sciona.atoms.ml.sklearn.naive_bayes import multinomial_nb_fit

    X, y = _training_data()
    expected = MultinomialNB(alpha=0.5).fit(X, y)
    state = multinomial_nb_fit(X, y, alpha=0.5)

    assert np.array_equal(state.classes, expected.classes_)
    assert np.allclose(state.class_count, expected.class_count_)
    assert np.allclose(state.feature_count, expected.feature_count_)
    assert np.allclose(state.class_log_prior, expected.class_log_prior_)
    assert np.allclose(state.feature_log_prob, expected.feature_log_prob_)


def test_multinomial_nb_fit_state_matches_sklearn_weighted_with_priors() -> None:
    from sciona.atoms.ml.sklearn.naive_bayes import multinomial_nb_fit

    X, y = _training_data()
    priors = np.array([0.4, 0.6], dtype=np.float64)
    weights = np.array([1.0, 1.5, 2.0, 1.25, 2.25, 1.75], dtype=np.float64)
    expected = MultinomialNB(alpha=1.25, class_prior=priors).fit(X, y, sample_weight=weights)
    state = multinomial_nb_fit(X, y, alpha=1.25, class_prior=priors, sample_weight=weights)

    assert np.array_equal(state.classes, expected.classes_)
    assert np.allclose(state.class_count, expected.class_count_)
    assert np.allclose(state.feature_count, expected.feature_count_)
    assert np.allclose(state.class_log_prior, expected.class_log_prior_)
    assert np.allclose(state.feature_log_prob, expected.feature_log_prob_)


def test_multinomial_nb_predictions_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.naive_bayes import (
        multinomial_nb_fit,
        multinomial_nb_joint_log_likelihood,
        multinomial_nb_predict,
        multinomial_nb_predict_log_proba,
        multinomial_nb_predict_proba,
    )

    X, y = _training_data()
    query = np.array([[2.0, 1.0, 0.0, 0.0], [0.0, 1.0, 3.0, 2.0], [1.0, 1.0, 1.0, 1.0]], dtype=np.float64)
    expected = MultinomialNB(alpha=0.75).fit(X, y)
    state = multinomial_nb_fit(X, y, alpha=0.75)

    assert np.allclose(multinomial_nb_joint_log_likelihood(query, state), expected._joint_log_likelihood(query))
    assert np.allclose(multinomial_nb_predict_log_proba(query, state), expected.predict_log_proba(query))
    assert np.allclose(multinomial_nb_predict_proba(query, state), expected.predict_proba(query))
    assert np.array_equal(multinomial_nb_predict(query, state), expected.predict(query))


def test_multinomial_nb_rejects_unsupported_inputs() -> None:
    from sciona.atoms.ml.sklearn.naive_bayes import multinomial_nb_fit, multinomial_nb_predict

    X, y = _training_data()
    state = multinomial_nb_fit(X, y)

    with pytest.raises(Exception):
        multinomial_nb_fit(-X, y)
    with pytest.raises(Exception):
        multinomial_nb_fit(X, np.zeros(X.shape[0], dtype=np.int64))
    with pytest.raises(Exception):
        multinomial_nb_fit(X, y, alpha=0.0)
    with pytest.raises(Exception):
        multinomial_nb_fit(X, y, class_prior=np.array([0.0, 1.0], dtype=np.float64))
    with pytest.raises(Exception):
        multinomial_nb_predict(np.ones((2, 2), dtype=np.float64), state)
