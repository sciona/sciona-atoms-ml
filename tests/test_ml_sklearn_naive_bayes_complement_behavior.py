from __future__ import annotations

import numpy as np
import pytest
from sklearn.naive_bayes import ComplementNB


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


def test_complement_nb_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.naive_bayes import (
        ComplementNBState,
        complement_nb_count,
        complement_nb_feature_log_prob,
        complement_nb_fit,
        complement_nb_joint_log_likelihood,
        complement_nb_predict,
        complement_nb_predict_log_proba,
        complement_nb_predict_proba,
    )

    assert ComplementNBState is not None
    assert callable(complement_nb_count)
    assert callable(complement_nb_feature_log_prob)
    assert callable(complement_nb_fit)
    assert callable(complement_nb_joint_log_likelihood)
    assert callable(complement_nb_predict)
    assert callable(complement_nb_predict_log_proba)
    assert callable(complement_nb_predict_proba)


def test_complement_nb_count_and_feature_weights_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.naive_bayes import (
        complement_nb_count,
        complement_nb_feature_log_prob,
    )

    X, y = _training_data()
    weights = np.array([1.0, 2.0, 1.5, 1.25, 1.75, 2.5], dtype=np.float64)
    expected = ComplementNB(alpha=0.75).fit(X, y, sample_weight=weights)
    classes, class_count, feature_count, feature_all = complement_nb_count(X, y, weights)
    feature_log_prob = complement_nb_feature_log_prob(feature_count, feature_all, alpha=0.75)

    assert np.array_equal(classes, expected.classes_)
    assert np.allclose(class_count, expected.class_count_)
    assert np.allclose(feature_count, expected.feature_count_)
    assert np.allclose(feature_all, expected.feature_all_)
    assert np.allclose(feature_log_prob, expected.feature_log_prob_)


def test_complement_nb_normalized_weights_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.naive_bayes import complement_nb_count, complement_nb_feature_log_prob

    X, y = _training_data()
    expected = ComplementNB(alpha=0.5, norm=True).fit(X, y)
    _, _, feature_count, feature_all = complement_nb_count(X, y)
    actual = complement_nb_feature_log_prob(feature_count, feature_all, alpha=0.5, norm=True)

    assert np.allclose(actual, expected.feature_log_prob_)


def test_complement_nb_fit_state_matches_sklearn_unweighted() -> None:
    from sciona.atoms.ml.sklearn.naive_bayes import complement_nb_fit

    X, y = _training_data()
    expected = ComplementNB(alpha=0.5).fit(X, y)
    state = complement_nb_fit(X, y, alpha=0.5)

    assert np.array_equal(state.classes, expected.classes_)
    assert np.allclose(state.class_count, expected.class_count_)
    assert np.allclose(state.feature_count, expected.feature_count_)
    assert np.allclose(state.feature_all, expected.feature_all_)
    assert np.allclose(state.class_log_prior, expected.class_log_prior_)
    assert np.allclose(state.feature_log_prob, expected.feature_log_prob_)


def test_complement_nb_fit_state_matches_sklearn_weighted_with_norm() -> None:
    from sciona.atoms.ml.sklearn.naive_bayes import complement_nb_fit

    X, y = _training_data()
    weights = np.array([1.0, 1.5, 2.0, 1.25, 2.25, 1.75], dtype=np.float64)
    expected = ComplementNB(alpha=1.25, norm=True).fit(X, y, sample_weight=weights)
    state = complement_nb_fit(X, y, alpha=1.25, norm=True, sample_weight=weights)

    assert np.array_equal(state.classes, expected.classes_)
    assert np.allclose(state.class_count, expected.class_count_)
    assert np.allclose(state.feature_count, expected.feature_count_)
    assert np.allclose(state.feature_all, expected.feature_all_)
    assert np.allclose(state.class_log_prior, expected.class_log_prior_)
    assert np.allclose(state.feature_log_prob, expected.feature_log_prob_)


def test_complement_nb_predictions_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.naive_bayes import (
        complement_nb_fit,
        complement_nb_joint_log_likelihood,
        complement_nb_predict,
        complement_nb_predict_log_proba,
        complement_nb_predict_proba,
    )

    X, y = _training_data()
    query = np.array([[2.0, 1.0, 0.0, 0.0], [0.0, 1.0, 3.0, 2.0], [1.0, 1.0, 1.0, 1.0]], dtype=np.float64)
    expected = ComplementNB(alpha=0.75).fit(X, y)
    state = complement_nb_fit(X, y, alpha=0.75)

    assert np.allclose(complement_nb_joint_log_likelihood(query, state), expected._joint_log_likelihood(query))
    assert np.allclose(complement_nb_predict_log_proba(query, state), expected.predict_log_proba(query))
    assert np.allclose(complement_nb_predict_proba(query, state), expected.predict_proba(query))
    assert np.array_equal(complement_nb_predict(query, state), expected.predict(query))


def test_complement_nb_rejects_unsupported_inputs() -> None:
    from sciona.atoms.ml.sklearn.naive_bayes import complement_nb_fit, complement_nb_predict

    X, y = _training_data()
    state = complement_nb_fit(X, y)

    with pytest.raises(Exception):
        complement_nb_fit(-X, y)
    with pytest.raises(Exception):
        complement_nb_fit(X, np.zeros(X.shape[0], dtype=np.int64))
    with pytest.raises(Exception):
        complement_nb_fit(X, y, alpha=0.0)
    with pytest.raises(Exception):
        complement_nb_fit(X, y, class_prior=np.array([0.0, 1.0], dtype=np.float64))
    with pytest.raises(Exception):
        complement_nb_predict(np.ones((2, 2), dtype=np.float64), state)
