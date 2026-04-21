from __future__ import annotations

import numpy as np
import pytest
from sklearn.naive_bayes import BernoulliNB


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


def test_bernoulli_nb_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.naive_bayes import (
        BernoulliNBState,
        bernoulli_nb_binarize,
        bernoulli_nb_count,
        bernoulli_nb_feature_log_prob,
        bernoulli_nb_fit,
        bernoulli_nb_joint_log_likelihood,
        bernoulli_nb_predict,
        bernoulli_nb_predict_log_proba,
        bernoulli_nb_predict_proba,
    )

    assert BernoulliNBState is not None
    assert callable(bernoulli_nb_binarize)
    assert callable(bernoulli_nb_count)
    assert callable(bernoulli_nb_feature_log_prob)
    assert callable(bernoulli_nb_fit)
    assert callable(bernoulli_nb_joint_log_likelihood)
    assert callable(bernoulli_nb_predict)
    assert callable(bernoulli_nb_predict_log_proba)
    assert callable(bernoulli_nb_predict_proba)


def test_bernoulli_nb_binarize_matches_sklearn_check_x() -> None:
    from sciona.atoms.ml.sklearn.naive_bayes import bernoulli_nb_binarize

    X, y = _training_data()
    expected = BernoulliNB(binarize=1.0).fit(X, y)

    assert np.array_equal(bernoulli_nb_binarize(X, binarize=1.0), expected._check_X(X))
    assert np.array_equal(bernoulli_nb_binarize((X > 0.0).astype(np.float64), binarize=None), (X > 0.0).astype(np.float64))


def test_bernoulli_nb_count_and_feature_log_prob_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.naive_bayes import bernoulli_nb_count, bernoulli_nb_feature_log_prob

    X, y = _training_data()
    weights = np.array([1.0, 2.0, 1.5, 1.25, 1.75, 2.5], dtype=np.float64)
    expected = BernoulliNB(alpha=0.75, binarize=0.5).fit(X, y, sample_weight=weights)
    classes, class_count, feature_count = bernoulli_nb_count(X, y, binarize=0.5, sample_weight=weights)
    feature_log_prob = bernoulli_nb_feature_log_prob(feature_count, class_count, alpha=0.75)

    assert np.array_equal(classes, expected.classes_)
    assert np.allclose(class_count, expected.class_count_)
    assert np.allclose(feature_count, expected.feature_count_)
    assert np.allclose(feature_log_prob, expected.feature_log_prob_)


def test_bernoulli_nb_fit_state_matches_sklearn_unweighted() -> None:
    from sciona.atoms.ml.sklearn.naive_bayes import bernoulli_nb_fit

    X, y = _training_data()
    expected = BernoulliNB(alpha=0.5, binarize=1.0).fit(X, y)
    state = bernoulli_nb_fit(X, y, alpha=0.5, binarize=1.0)

    assert np.array_equal(state.classes, expected.classes_)
    assert np.allclose(state.class_count, expected.class_count_)
    assert np.allclose(state.feature_count, expected.feature_count_)
    assert np.allclose(state.class_log_prior, expected.class_log_prior_)
    assert np.allclose(state.feature_log_prob, expected.feature_log_prob_)


def test_bernoulli_nb_fit_state_matches_sklearn_weighted_with_class_prior() -> None:
    from sciona.atoms.ml.sklearn.naive_bayes import bernoulli_nb_fit

    X, y = _training_data()
    weights = np.array([1.0, 1.5, 2.0, 1.25, 2.25, 1.75], dtype=np.float64)
    class_prior = np.array([0.35, 0.65], dtype=np.float64)
    expected = BernoulliNB(alpha=1.25, binarize=0.0, class_prior=class_prior).fit(X, y, sample_weight=weights)
    state = bernoulli_nb_fit(X, y, alpha=1.25, binarize=0.0, class_prior=class_prior, sample_weight=weights)

    assert np.array_equal(state.classes, expected.classes_)
    assert np.allclose(state.class_count, expected.class_count_)
    assert np.allclose(state.feature_count, expected.feature_count_)
    assert np.allclose(state.class_log_prior, expected.class_log_prior_)
    assert np.allclose(state.feature_log_prob, expected.feature_log_prob_)


def test_bernoulli_nb_predictions_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.naive_bayes import (
        bernoulli_nb_fit,
        bernoulli_nb_joint_log_likelihood,
        bernoulli_nb_predict,
        bernoulli_nb_predict_log_proba,
        bernoulli_nb_predict_proba,
    )

    X, y = _training_data()
    query = np.array([[2.0, 1.0, 0.0, 0.0], [0.0, 1.0, 3.0, 2.0], [1.0, 1.0, 1.0, 1.0]], dtype=np.float64)
    expected = BernoulliNB(alpha=0.75, binarize=1.0).fit(X, y)
    state = bernoulli_nb_fit(X, y, alpha=0.75, binarize=1.0)

    assert np.allclose(bernoulli_nb_joint_log_likelihood(query, state), expected._joint_log_likelihood(expected._check_X(query)))
    assert np.allclose(bernoulli_nb_predict_log_proba(query, state), expected.predict_log_proba(query))
    assert np.allclose(bernoulli_nb_predict_proba(query, state), expected.predict_proba(query))
    assert np.array_equal(bernoulli_nb_predict(query, state), expected.predict(query))


def test_bernoulli_nb_binarize_none_predictions_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.naive_bayes import bernoulli_nb_fit, bernoulli_nb_predict_log_proba

    X, y = _training_data()
    binary = (X > 0.0).astype(np.float64)
    expected = BernoulliNB(alpha=0.75, binarize=None).fit(binary, y)
    state = bernoulli_nb_fit(binary, y, alpha=0.75, binarize=None)

    assert np.allclose(bernoulli_nb_predict_log_proba(binary, state), expected.predict_log_proba(binary))


def test_bernoulli_nb_rejects_unsupported_inputs() -> None:
    from sciona.atoms.ml.sklearn.naive_bayes import bernoulli_nb_fit, bernoulli_nb_predict

    X, y = _training_data()
    state = bernoulli_nb_fit(X, y)

    with pytest.raises(Exception):
        bernoulli_nb_fit(X, y, binarize=-1.0)
    with pytest.raises(Exception):
        bernoulli_nb_fit(X, y, binarize=None)
    with pytest.raises(Exception):
        bernoulli_nb_fit(X, np.zeros(X.shape[0], dtype=np.int64))
    with pytest.raises(Exception):
        bernoulli_nb_fit(X, y, alpha=0.0)
    with pytest.raises(Exception):
        bernoulli_nb_fit(X, y, class_prior=np.array([0.0, 1.0], dtype=np.float64))
    with pytest.raises(Exception):
        bernoulli_nb_predict(np.ones((2, 2), dtype=np.float64), state)
