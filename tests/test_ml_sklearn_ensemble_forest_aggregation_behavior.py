from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def test_forest_aggregation_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_aggregation import (
        forest_classifier_average_probabilities,
        forest_classifier_labels_from_probabilities,
        forest_regressor_average_predictions,
    )

    assert callable(forest_classifier_average_probabilities)
    assert callable(forest_classifier_labels_from_probabilities)
    assert callable(forest_regressor_average_predictions)


def test_forest_classifier_average_probabilities_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_aggregation import forest_classifier_average_probabilities

    X, y = make_classification(
        n_samples=90,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=7,
    )
    clf = RandomForestClassifier(n_estimators=9, random_state=3)
    clf.fit(X, y)

    probabilities = np.stack([estimator.predict_proba(X) for estimator in clf.estimators_], axis=0).astype(np.float64)
    result = forest_classifier_average_probabilities(probabilities)
    expected = clf.predict_proba(X)
    assert np.allclose(result, expected)


def test_forest_classifier_labels_from_probabilities_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_aggregation import (
        forest_classifier_average_probabilities,
        forest_classifier_labels_from_probabilities,
    )

    X, y = make_classification(
        n_samples=80,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        random_state=11,
    )
    clf = RandomForestClassifier(n_estimators=7, random_state=5)
    clf.fit(X, y)

    probabilities = np.stack([estimator.predict_proba(X) for estimator in clf.estimators_], axis=0).astype(np.float64)
    averaged = forest_classifier_average_probabilities(probabilities)
    result = forest_classifier_labels_from_probabilities(averaged, clf.classes_.astype(np.float64))
    expected = clf.predict(X).astype(np.float64)
    assert np.array_equal(result, expected)


def test_forest_regressor_average_predictions_matches_sklearn_single_output() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_aggregation import forest_regressor_average_predictions

    X, y = make_regression(n_samples=75, n_features=5, noise=0.2, random_state=13)
    reg = RandomForestRegressor(n_estimators=8, random_state=2)
    reg.fit(X, y)

    predictions = np.stack([estimator.predict(X) for estimator in reg.estimators_], axis=0).astype(np.float64)
    result = forest_regressor_average_predictions(predictions)
    expected = reg.predict(X)
    assert np.allclose(result, expected)


def test_forest_regressor_average_predictions_matches_sklearn_multioutput() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_aggregation import forest_regressor_average_predictions

    X, y1 = make_regression(n_samples=70, n_features=4, noise=0.1, random_state=17)
    _, y2 = make_regression(n_samples=70, n_features=4, noise=0.3, random_state=19)
    y = np.column_stack([y1, y2]).astype(np.float64)
    reg = RandomForestRegressor(n_estimators=6, random_state=4)
    reg.fit(X, y)

    predictions = np.stack([estimator.predict(X) for estimator in reg.estimators_], axis=0).astype(np.float64)
    result = forest_regressor_average_predictions(predictions)
    expected = reg.predict(X)
    assert np.allclose(result, expected)


def test_contracts_reject_invalid_forest_aggregation_inputs() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_aggregation import (
        forest_classifier_average_probabilities,
        forest_classifier_labels_from_probabilities,
        forest_regressor_average_predictions,
    )

    with pytest.raises(ViolationError):
        forest_classifier_average_probabilities(
            np.array([[[0.2, 0.5], [0.4, 0.7]]], dtype=np.float64),
        )

    with pytest.raises(ViolationError):
        forest_classifier_labels_from_probabilities(
            np.array([[0.4, 0.6], [0.2, 0.8]], dtype=np.float64),
            np.array([0.0, 0.0], dtype=np.float64),
        )

    with pytest.raises(ViolationError):
        forest_regressor_average_predictions(np.array([1.0, 2.0, 3.0], dtype=np.float64))
