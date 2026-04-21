from __future__ import annotations

import numpy as np
import pytest
from sklearn.naive_bayes import CategoricalNB


def _training_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [0, 1, 2],
            [1, 0, 1],
            [2, 1, 0],
            [0, 2, 1],
            [1, 2, 2],
            [2, 0, 0],
        ],
        dtype=np.int64,
    )
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    return X, y


def test_categorical_nb_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.naive_bayes import (
        CategoricalNBState,
        categorical_nb_count,
        categorical_nb_feature_log_prob,
        categorical_nb_fit,
        categorical_nb_joint_log_likelihood,
        categorical_nb_n_categories,
        categorical_nb_predict,
        categorical_nb_predict_log_proba,
        categorical_nb_predict_proba,
    )

    assert CategoricalNBState is not None
    assert callable(categorical_nb_n_categories)
    assert callable(categorical_nb_count)
    assert callable(categorical_nb_feature_log_prob)
    assert callable(categorical_nb_fit)
    assert callable(categorical_nb_joint_log_likelihood)
    assert callable(categorical_nb_predict)
    assert callable(categorical_nb_predict_log_proba)
    assert callable(categorical_nb_predict_proba)


def test_categorical_nb_category_counts_match_sklearn_weighted() -> None:
    from sciona.atoms.ml.sklearn.naive_bayes import categorical_nb_count, categorical_nb_feature_log_prob

    X, y = _training_data()
    weights = np.array([1.0, 1.5, 2.0, 1.25, 2.25, 1.75], dtype=np.float64)
    min_categories = np.array([3, 4, 5], dtype=np.int64)
    expected = CategoricalNB(alpha=0.75, min_categories=min_categories).fit(X, y, sample_weight=weights)
    classes, class_count, n_categories, category_count = categorical_nb_count(X, y, min_categories, weights)
    feature_log_prob = categorical_nb_feature_log_prob(category_count, alpha=0.75)

    assert np.array_equal(classes, expected.classes_)
    assert np.allclose(class_count, expected.class_count_)
    assert np.array_equal(n_categories, expected.n_categories_)
    assert all(np.allclose(actual, target) for actual, target in zip(category_count, expected.category_count_))
    assert all(np.allclose(actual, target) for actual, target in zip(feature_log_prob, expected.feature_log_prob_))


def test_categorical_nb_n_categories_matches_sklearn_scalar_minimum() -> None:
    from sciona.atoms.ml.sklearn.naive_bayes import categorical_nb_n_categories

    X, y = _training_data()
    expected = CategoricalNB(min_categories=4).fit(X, y)

    assert np.array_equal(categorical_nb_n_categories(X, 4), expected.n_categories_)


def test_categorical_nb_fit_state_matches_sklearn_unweighted() -> None:
    from sciona.atoms.ml.sklearn.naive_bayes import categorical_nb_fit

    X, y = _training_data()
    expected = CategoricalNB(alpha=0.5).fit(X, y)
    state = categorical_nb_fit(X, y, alpha=0.5)

    assert np.array_equal(state.classes, expected.classes_)
    assert np.allclose(state.class_count, expected.class_count_)
    assert np.array_equal(state.n_categories, expected.n_categories_)
    assert np.allclose(state.class_log_prior, expected.class_log_prior_)
    assert all(np.allclose(actual, target) for actual, target in zip(state.category_count, expected.category_count_))
    assert all(np.allclose(actual, target) for actual, target in zip(state.feature_log_prob, expected.feature_log_prob_))


def test_categorical_nb_fit_state_matches_sklearn_weighted_with_prior() -> None:
    from sciona.atoms.ml.sklearn.naive_bayes import categorical_nb_fit

    X, y = _training_data()
    weights = np.array([1.0, 1.5, 2.0, 1.25, 2.25, 1.75], dtype=np.float64)
    class_prior = np.array([0.4, 0.6], dtype=np.float64)
    expected = CategoricalNB(alpha=1.25, class_prior=class_prior, min_categories=4).fit(X, y, sample_weight=weights)
    state = categorical_nb_fit(X, y, alpha=1.25, class_prior=class_prior, min_categories=4, sample_weight=weights)

    assert np.array_equal(state.classes, expected.classes_)
    assert np.allclose(state.class_count, expected.class_count_)
    assert np.array_equal(state.n_categories, expected.n_categories_)
    assert np.allclose(state.class_log_prior, expected.class_log_prior_)
    assert all(np.allclose(actual, target) for actual, target in zip(state.category_count, expected.category_count_))
    assert all(np.allclose(actual, target) for actual, target in zip(state.feature_log_prob, expected.feature_log_prob_))


def test_categorical_nb_predictions_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.naive_bayes import (
        categorical_nb_fit,
        categorical_nb_joint_log_likelihood,
        categorical_nb_predict,
        categorical_nb_predict_log_proba,
        categorical_nb_predict_proba,
    )

    X, y = _training_data()
    query = np.array([[0, 1, 2], [2, 0, 0], [1, 2, 1]], dtype=np.int64)
    expected = CategoricalNB(alpha=0.75, min_categories=np.array([3, 4, 5], dtype=np.int64)).fit(X, y)
    state = categorical_nb_fit(X, y, alpha=0.75, min_categories=np.array([3, 4, 5], dtype=np.int64))

    assert np.allclose(categorical_nb_joint_log_likelihood(query, state), expected._joint_log_likelihood(query))
    assert np.allclose(categorical_nb_predict_log_proba(query, state), expected.predict_log_proba(query))
    assert np.allclose(categorical_nb_predict_proba(query, state), expected.predict_proba(query))
    assert np.array_equal(categorical_nb_predict(query, state), expected.predict(query))


def test_categorical_nb_rejects_unsupported_inputs() -> None:
    from sciona.atoms.ml.sklearn.naive_bayes import categorical_nb_fit, categorical_nb_predict

    X, y = _training_data()
    state = categorical_nb_fit(X, y)

    with pytest.raises(Exception):
        categorical_nb_fit(X.astype(np.float64), y)
    with pytest.raises(Exception):
        categorical_nb_fit(-X, y)
    with pytest.raises(Exception):
        categorical_nb_fit(X, np.zeros(X.shape[0], dtype=np.int64))
    with pytest.raises(Exception):
        categorical_nb_fit(X, y, alpha=0.0)
    with pytest.raises(Exception):
        categorical_nb_fit(X, y, min_categories=np.array([3, 4], dtype=np.int64))
    with pytest.raises(Exception):
        categorical_nb_fit(X, y, class_prior=np.array([0.0, 1.0], dtype=np.float64))
    with pytest.raises(Exception):
        categorical_nb_predict(np.array([[3, 0, 0]], dtype=np.int64), state)
