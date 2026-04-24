from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def test_bagging_aggregation_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_aggregation import (
        bagging_classifier_average_decision_function,
        bagging_classifier_average_log_probabilities,
        bagging_classifier_average_probabilities,
        bagging_regressor_average_predictions,
    )

    assert callable(bagging_classifier_average_probabilities)
    assert callable(bagging_classifier_average_log_probabilities)
    assert callable(bagging_classifier_average_decision_function)
    assert callable(bagging_regressor_average_predictions)


def test_bagging_classifier_average_probabilities_aligns_missing_classes() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_aggregation import bagging_classifier_average_probabilities

    probability_blocks = (
        np.array([[0.2, 0.5, 0.3], [0.6, 0.1, 0.3]], dtype=np.float64),
        np.array([[0.7, 0.3], [0.4, 0.6]], dtype=np.float64),
    )
    class_index_blocks = (
        np.array([0, 1, 2], dtype=np.int64),
        np.array([0, 2], dtype=np.int64),
    )

    result = bagging_classifier_average_probabilities(
        probability_blocks,
        class_index_blocks,
        n_classes=3,
    )

    expected = np.array([[0.45, 0.25, 0.30], [0.50, 0.05, 0.45]], dtype=np.float64)
    assert np.allclose(result, expected)


def test_bagging_classifier_average_probabilities_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_aggregation import bagging_classifier_average_probabilities

    X, y = make_classification(
        n_samples=120,
        n_features=6,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        weights=[0.65, 0.25, 0.10],
        random_state=7,
    )
    clf = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=0),
        n_estimators=9,
        max_samples=0.35,
        max_features=1.0,
        bootstrap=True,
        random_state=5,
    )
    clf.fit(X, y)

    probability_blocks = tuple(
        estimator.predict_proba(X[:, features]).astype(np.float64)
        for estimator, features in zip(clf.estimators_, clf.estimators_features_)
    )
    class_index_blocks = tuple(
        np.searchsorted(clf.classes_, estimator.classes_).astype(np.int64)
        for estimator in clf.estimators_
    )

    result = bagging_classifier_average_probabilities(
        probability_blocks,
        class_index_blocks,
        n_classes=int(clf.classes_.shape[0]),
    )
    assert np.allclose(result, clf.predict_proba(X))


def test_bagging_classifier_average_log_probabilities_aligns_missing_classes() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_aggregation import bagging_classifier_average_log_probabilities

    probability_blocks = (
        np.array([[0.25, 0.75], [0.8, 0.2]], dtype=np.float64),
        np.array([[0.5, 0.1, 0.4], [0.3, 0.4, 0.3]], dtype=np.float64),
    )
    log_probability_blocks = tuple(np.log(block) for block in probability_blocks)
    class_index_blocks = (
        np.array([0, 2], dtype=np.int64),
        np.array([0, 1, 2], dtype=np.int64),
    )

    result = bagging_classifier_average_log_probabilities(
        log_probability_blocks,
        class_index_blocks,
        n_classes=3,
    )

    expected_probabilities = np.array([[0.375, 0.05, 0.575], [0.55, 0.2, 0.25]], dtype=np.float64)
    assert np.allclose(np.exp(result), expected_probabilities)


def test_bagging_classifier_average_log_probabilities_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_aggregation import bagging_classifier_average_log_probabilities

    X, y = make_classification(
        n_samples=110,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        weights=[0.6, 0.3, 0.1],
        random_state=13,
    )
    clf = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=0),
        n_estimators=7,
        max_samples=0.4,
        max_features=1.0,
        bootstrap=True,
        random_state=11,
    )
    clf.fit(X, y)

    log_probability_blocks = tuple(
        estimator.predict_log_proba(X[:, features]).astype(np.float64)
        for estimator, features in zip(clf.estimators_, clf.estimators_features_)
    )
    class_index_blocks = tuple(
        np.searchsorted(clf.classes_, estimator.classes_).astype(np.int64)
        for estimator in clf.estimators_
    )

    result = bagging_classifier_average_log_probabilities(
        log_probability_blocks,
        class_index_blocks,
        n_classes=int(clf.classes_.shape[0]),
    )
    assert np.allclose(result, clf.predict_log_proba(X))


def test_bagging_classifier_average_decision_function_matches_mean() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_aggregation import bagging_classifier_average_decision_function

    decision_blocks = (
        np.array([[1.0, -0.5], [0.2, 0.8]], dtype=np.float64),
        np.array([[0.0, 0.5], [0.4, 1.2]], dtype=np.float64),
        np.array([[2.0, -1.0], [-0.2, 0.4]], dtype=np.float64),
    )

    result = bagging_classifier_average_decision_function(decision_blocks)
    assert np.allclose(result, np.mean(np.stack(decision_blocks, axis=0), axis=0))


def test_bagging_classifier_average_decision_function_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_aggregation import bagging_classifier_average_decision_function

    X, y = make_classification(
        n_samples=100,
        n_features=6,
        n_informative=5,
        n_redundant=0,
        random_state=19,
    )
    clf = BaggingClassifier(
        estimator=LogisticRegression(max_iter=500),
        n_estimators=5,
        max_samples=1.0,
        max_features=0.75,
        bootstrap=False,
        random_state=17,
    )
    clf.fit(X, y)

    decision_blocks = tuple(
        np.asarray(estimator.decision_function(X[:, features]), dtype=np.float64)
        for estimator, features in zip(clf.estimators_, clf.estimators_features_)
    )
    result = bagging_classifier_average_decision_function(decision_blocks)
    assert np.allclose(result, clf.decision_function(X))


def test_bagging_regressor_average_predictions_supports_multioutput() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_aggregation import bagging_regressor_average_predictions

    prediction_blocks = (
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        np.array([[2.0, 0.0], [1.0, 5.0]], dtype=np.float64),
    )

    result = bagging_regressor_average_predictions(prediction_blocks)
    expected = np.array([[1.5, 1.0], [2.0, 4.5]], dtype=np.float64)
    assert np.allclose(result, expected)


def test_bagging_regressor_average_predictions_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_aggregation import bagging_regressor_average_predictions

    X, y = make_regression(n_samples=90, n_features=5, noise=0.3, random_state=23)
    reg = BaggingRegressor(
        estimator=DecisionTreeRegressor(random_state=0),
        n_estimators=8,
        max_samples=0.8,
        max_features=0.8,
        bootstrap=True,
        random_state=29,
    )
    reg.fit(X, y)

    prediction_blocks = tuple(
        np.asarray(estimator.predict(X[:, features]), dtype=np.float64)
        for estimator, features in zip(reg.estimators_, reg.estimators_features_)
    )
    result = bagging_regressor_average_predictions(prediction_blocks)
    assert np.allclose(result, reg.predict(X))


def test_contracts_reject_invalid_bagging_aggregation_inputs() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_aggregation import (
        bagging_classifier_average_decision_function,
        bagging_classifier_average_log_probabilities,
        bagging_classifier_average_probabilities,
        bagging_regressor_average_predictions,
    )

    with pytest.raises(ViolationError):
        bagging_classifier_average_probabilities(
            (np.array([[0.5, 0.5]], dtype=np.float64),),
            (np.array([0, 3], dtype=np.int64),),
            n_classes=3,
        )

    with pytest.raises(ViolationError):
        bagging_classifier_average_log_probabilities(
            (np.array([[0.0, -1.0]], dtype=np.float64),),
            (np.array([0, 1], dtype=np.int64),),
            n_classes=2,
        )

    with pytest.raises(ViolationError):
        bagging_classifier_average_decision_function(
            (
                np.array([1.0, 2.0], dtype=np.float64),
                np.array([[1.0, 2.0]], dtype=np.float64),
            )
        )

    with pytest.raises(ViolationError):
        bagging_regressor_average_predictions(tuple())
