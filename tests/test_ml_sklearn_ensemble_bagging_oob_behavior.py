from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def _oob_rows(sample_indices: np.ndarray, n_samples: int) -> np.ndarray:
    in_bag = np.zeros(n_samples, dtype=bool)
    in_bag[np.asarray(sample_indices, dtype=np.int64)] = True
    return np.flatnonzero(~in_bag)


def test_bagging_oob_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_oob import (
        bagging_classifier_oob_decision_function,
        bagging_classifier_oob_label_indices,
        bagging_classifier_oob_probability_totals,
        bagging_classifier_oob_vote_totals,
        bagging_regressor_oob_predictions,
    )

    assert callable(bagging_classifier_oob_probability_totals)
    assert callable(bagging_classifier_oob_vote_totals)
    assert callable(bagging_classifier_oob_decision_function)
    assert callable(bagging_classifier_oob_label_indices)
    assert callable(bagging_regressor_oob_predictions)


def test_bagging_classifier_oob_probability_totals_align_missing_classes() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_oob import bagging_classifier_oob_probability_totals

    probability_blocks = (
        np.array([[0.2, 0.8], [0.7, 0.3]], dtype=np.float64),
        np.array([[0.5, 0.5]], dtype=np.float64),
    )
    class_index_blocks = (
        np.array([0, 2], dtype=np.int64),
        np.array([1, 2], dtype=np.int64),
    )
    sample_index_blocks = (
        np.array([0, 1], dtype=np.int64),
        np.array([0, 1, 2], dtype=np.int64),
    )

    result = bagging_classifier_oob_probability_totals(
        probability_blocks,
        class_index_blocks,
        sample_index_blocks,
        n_samples=4,
        n_classes=3,
    )

    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.8],
            [0.7, 0.5, 0.8],
        ],
        dtype=np.float64,
    )
    assert np.allclose(result, expected)


def test_bagging_classifier_oob_probability_path_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_oob import (
        bagging_classifier_oob_decision_function,
        bagging_classifier_oob_label_indices,
        bagging_classifier_oob_probability_totals,
    )

    X, y = make_classification(
        n_samples=120,
        n_features=6,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        weights=[0.7, 0.2, 0.1],
        random_state=7,
    )
    clf = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=0),
        n_estimators=25,
        bootstrap=True,
        oob_score=True,
        random_state=11,
    )
    clf.fit(X, y)

    sample_index_blocks = tuple(np.asarray(samples, dtype=np.int64) for samples in clf.estimators_samples_)
    probability_blocks = []
    class_index_blocks = []
    for estimator, sample_indices, features in zip(clf.estimators_, sample_index_blocks, clf.estimators_features_):
        rows = _oob_rows(sample_indices, X.shape[0])
        probability_blocks.append(np.asarray(estimator.predict_proba(X[rows][:, features]), dtype=np.float64))
        class_index_blocks.append(np.searchsorted(clf.classes_, estimator.classes_).astype(np.int64))

    totals = bagging_classifier_oob_probability_totals(
        tuple(probability_blocks),
        tuple(class_index_blocks),
        sample_index_blocks,
        n_samples=X.shape[0],
        n_classes=int(clf.classes_.shape[0]),
    )
    decision = bagging_classifier_oob_decision_function(totals)
    labels = bagging_classifier_oob_label_indices(totals)

    assert np.allclose(decision, clf.oob_decision_function_, equal_nan=True)
    assert np.array_equal(labels, np.argmax(np.nan_to_num(clf.oob_decision_function_, nan=0.0), axis=1))


def test_bagging_classifier_oob_vote_path_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_oob import (
        bagging_classifier_oob_decision_function,
        bagging_classifier_oob_label_indices,
        bagging_classifier_oob_vote_totals,
    )

    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        weights=[0.6, 0.3, 0.1],
        random_state=13,
    )
    clf = BaggingClassifier(
        estimator=LinearSVC(dual="auto", random_state=0),
        n_estimators=25,
        bootstrap=True,
        oob_score=True,
        random_state=17,
    )
    clf.fit(X, y)

    sample_index_blocks = tuple(np.asarray(samples, dtype=np.int64) for samples in clf.estimators_samples_)
    predicted_label_blocks = tuple(
        np.asarray(estimator.predict(X[_oob_rows(sample_indices, X.shape[0])][:, features]), dtype=np.int64)
        for estimator, sample_indices, features in zip(clf.estimators_, sample_index_blocks, clf.estimators_features_)
    )

    totals = bagging_classifier_oob_vote_totals(
        predicted_label_blocks,
        sample_index_blocks,
        n_samples=X.shape[0],
        n_classes=int(clf.classes_.shape[0]),
    )
    decision = bagging_classifier_oob_decision_function(totals)
    labels = bagging_classifier_oob_label_indices(totals)

    assert np.allclose(decision, clf.oob_decision_function_, equal_nan=True)
    assert np.array_equal(labels, np.argmax(np.nan_to_num(clf.oob_decision_function_, nan=0.0), axis=1))


def test_bagging_regressor_oob_predictions_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_oob import bagging_regressor_oob_predictions

    X, y = make_regression(n_samples=110, n_features=5, noise=0.4, random_state=19)
    reg = BaggingRegressor(
        estimator=DecisionTreeRegressor(random_state=0),
        n_estimators=20,
        bootstrap=True,
        oob_score=True,
        random_state=23,
    )
    reg.fit(X, y)

    sample_index_blocks = tuple(np.asarray(samples, dtype=np.int64) for samples in reg.estimators_samples_)
    prediction_blocks = tuple(
        np.asarray(estimator.predict(X[_oob_rows(sample_indices, X.shape[0])][:, features]), dtype=np.float64)
        for estimator, sample_indices, features in zip(reg.estimators_, sample_index_blocks, reg.estimators_features_)
    )

    result = bagging_regressor_oob_predictions(
        prediction_blocks,
        sample_index_blocks,
        n_samples=X.shape[0],
    )
    assert np.allclose(result, reg.oob_prediction_)


def test_contracts_reject_invalid_bagging_oob_inputs() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_oob import (
        bagging_classifier_oob_decision_function,
        bagging_classifier_oob_probability_totals,
        bagging_classifier_oob_vote_totals,
        bagging_regressor_oob_predictions,
    )

    with pytest.raises(ViolationError):
        bagging_classifier_oob_probability_totals(
            (np.array([[0.5, 0.5]], dtype=np.float64),),
            (np.array([0, 3], dtype=np.int64),),
            (np.array([0], dtype=np.int64),),
            n_samples=3,
            n_classes=3,
        )

    with pytest.raises(ViolationError):
        bagging_classifier_oob_vote_totals(
            (np.array([0, 4], dtype=np.int64),),
            (np.array([0], dtype=np.int64),),
            n_samples=3,
            n_classes=3,
        )

    with pytest.raises(ViolationError):
        bagging_classifier_oob_decision_function(np.array([[1.0, -1.0]], dtype=np.float64))

    with pytest.raises(ViolationError):
        bagging_regressor_oob_predictions(
            (np.array([1.0, 2.0], dtype=np.float64),),
            (np.array([0, 1], dtype=np.int64),),
            n_samples=3,
        )
