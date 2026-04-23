from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor


def _encoded_estimator_predictions(clf: AdaBoostClassifier, X: np.ndarray) -> np.ndarray:
    classes = clf.classes_.astype(np.float64)
    encoded_columns = [
        np.searchsorted(classes, estimator.predict(X).astype(np.float64)).astype(np.int64)
        for estimator in clf.estimators_
    ]
    return np.column_stack(encoded_columns)


def test_adaboost_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.ensemble.adaboost import (
        adaboost_classifier_decision_function,
        adaboost_classifier_probabilities_from_decision,
        adaboost_regressor_weighted_median,
    )

    assert callable(adaboost_classifier_decision_function)
    assert callable(adaboost_classifier_probabilities_from_decision)
    assert callable(adaboost_regressor_weighted_median)


@pytest.mark.parametrize("n_classes", [2, 3])
def test_adaboost_classifier_decision_matches_sklearn(n_classes: int) -> None:
    from sciona.atoms.ml.sklearn.ensemble.adaboost import adaboost_classifier_decision_function

    X, y = make_classification(
        n_samples=80,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=7,
    )
    clf = AdaBoostClassifier(n_estimators=8, random_state=3)
    clf.fit(X, y)

    encoded_predictions = _encoded_estimator_predictions(clf, X)
    result = adaboost_classifier_decision_function(
        encoded_predictions,
        clf.classes_.astype(np.float64),
        tuple(float(w) for w in clf.estimator_weights_[: len(clf.estimators_)]),
    )
    expected = clf.decision_function(X)
    assert np.allclose(result, expected)


@pytest.mark.parametrize("n_classes", [2, 3])
def test_adaboost_classifier_probabilities_match_sklearn(n_classes: int) -> None:
    from sciona.atoms.ml.sklearn.ensemble.adaboost import (
        adaboost_classifier_decision_function,
        adaboost_classifier_probabilities_from_decision,
    )

    X, y = make_classification(
        n_samples=70,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=11,
    )
    clf = AdaBoostClassifier(n_estimators=7, random_state=5)
    clf.fit(X, y)

    decision = adaboost_classifier_decision_function(
        _encoded_estimator_predictions(clf, X),
        clf.classes_.astype(np.float64),
        tuple(float(w) for w in clf.estimator_weights_[: len(clf.estimators_)]),
    )
    result = adaboost_classifier_probabilities_from_decision(decision, int(clf.n_classes_))
    expected = clf.predict_proba(X)
    assert np.allclose(result, expected)


def test_adaboost_regressor_weighted_median_matches_sklearn_predict() -> None:
    from sciona.atoms.ml.sklearn.ensemble.adaboost import adaboost_regressor_weighted_median

    X, y = make_regression(n_samples=90, n_features=5, noise=0.3, random_state=13)
    reg = AdaBoostRegressor(n_estimators=9, random_state=2)
    reg.fit(X, y)

    predictions = np.column_stack([estimator.predict(X) for estimator in reg.estimators_]).astype(np.float64)
    result = adaboost_regressor_weighted_median(
        predictions,
        tuple(float(w) for w in reg.estimator_weights_[: len(reg.estimators_)]),
    )
    expected = reg.predict(X)
    assert np.allclose(result, expected)


def test_contracts_reject_invalid_adaboost_inputs() -> None:
    from sciona.atoms.ml.sklearn.ensemble.adaboost import (
        adaboost_classifier_decision_function,
        adaboost_classifier_probabilities_from_decision,
        adaboost_regressor_weighted_median,
    )

    with pytest.raises(ViolationError):
        adaboost_classifier_decision_function(
            np.array([[0, 1], [1, 2]], dtype=np.int64),
            np.array([0.0, 1.0], dtype=np.float64),
            (1.0, 1.0),
        )

    with pytest.raises(ViolationError):
        adaboost_classifier_decision_function(
            np.array([[0, 1], [1, 0]], dtype=np.int64),
            np.array([0.0, 1.0], dtype=np.float64),
            (0.0, 0.0),
        )

    with pytest.raises(ViolationError):
        adaboost_classifier_probabilities_from_decision(np.array([[0.1, 0.2]], dtype=np.float64), 2)

    with pytest.raises(ViolationError):
        adaboost_regressor_weighted_median(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            (1.0,),
        )
