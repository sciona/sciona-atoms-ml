from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.preprocessing import LabelEncoder


def test_voting_aggregation_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.ensemble.voting import (
        voting_classifier_hard_labels,
        voting_classifier_soft_probabilities,
        voting_regressor_average,
    )

    assert callable(voting_classifier_hard_labels)
    assert callable(voting_classifier_soft_probabilities)
    assert callable(voting_regressor_average)


def test_hard_labels_match_sklearn_weighted_vote_rule() -> None:
    from sciona.atoms.ml.sklearn.ensemble.voting import voting_classifier_hard_labels

    encoded_predictions = np.array(
        [
            [0, 1, 1],
            [2, 2, 1],
            [1, 0, 2],
            [0, 2, 1],
        ],
        dtype=np.int64,
    )
    classes = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    weights = (0.2, 1.0, 0.6)

    voter = VotingClassifier(
        estimators=[("a", object()), ("b", object()), ("c", object())],
        voting="hard",
        weights=list(weights),
    )
    voter.le_ = LabelEncoder().fit(classes)
    voter.classes_ = voter.le_.classes_
    voter.estimators_ = [object(), object(), object()]
    voter._predict = lambda X: encoded_predictions

    result = voting_classifier_hard_labels(encoded_predictions, classes, weights=weights)
    expected = voter.predict(np.zeros((encoded_predictions.shape[0], 1)))
    assert np.array_equal(result, expected)


def test_hard_labels_use_lowest_encoded_class_for_ties() -> None:
    from sciona.atoms.ml.sklearn.ensemble.voting import voting_classifier_hard_labels

    encoded_predictions = np.array([[0, 1], [1, 0]], dtype=np.int64)
    classes = np.array([4.0, 9.0], dtype=np.float64)
    assert np.array_equal(voting_classifier_hard_labels(encoded_predictions, classes), np.array([4.0, 4.0]))


def test_soft_probabilities_match_sklearn_predict_proba_average() -> None:
    from sciona.atoms.ml.sklearn.ensemble.voting import voting_classifier_soft_probabilities

    probabilities = np.array(
        [
            [[0.7, 0.2, 0.1], [0.1, 0.7, 0.2]],
            [[0.6, 0.1, 0.3], [0.3, 0.4, 0.3]],
            [[0.2, 0.4, 0.4], [0.2, 0.3, 0.5]],
        ],
        dtype=np.float64,
    )
    weights = (0.5, 1.0, 1.5)

    voter = VotingClassifier(
        estimators=[("a", object()), ("b", object()), ("c", object())],
        voting="soft",
        weights=list(weights),
    )
    voter.classes_ = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    voter.estimators_ = [object(), object(), object()]
    voter._collect_probas = lambda X: probabilities

    result = voting_classifier_soft_probabilities(probabilities, weights=weights)
    expected = voter.predict_proba(np.zeros((probabilities.shape[1], 1)))
    assert np.allclose(result, expected)
    assert np.allclose(result.sum(axis=1), 1.0)


def test_regressor_average_matches_sklearn_prediction_average() -> None:
    from sciona.atoms.ml.sklearn.ensemble.voting import voting_regressor_average

    predictions = np.array(
        [
            [1.0, 2.0, 4.0],
            [3.0, 4.0, 8.0],
            [10.0, 8.0, 2.0],
        ],
        dtype=np.float64,
    )
    weights = (1.0, 2.0, 1.0)

    voter = VotingRegressor(
        estimators=[("a", object()), ("b", object()), ("c", object())],
        weights=list(weights),
    )
    voter.estimators_ = [object(), object(), object()]
    voter._predict = lambda X: predictions

    result = voting_regressor_average(predictions, weights=weights)
    expected = voter.predict(np.zeros((predictions.shape[0], 1)))
    assert np.allclose(result, expected)


def test_contracts_reject_invalid_probabilities_and_weights() -> None:
    from sciona.atoms.ml.sklearn.ensemble.voting import (
        voting_classifier_soft_probabilities,
        voting_regressor_average,
    )

    bad_probabilities = np.array([[[0.6, 0.6]]], dtype=np.float64)
    with pytest.raises(ViolationError):
        voting_classifier_soft_probabilities(bad_probabilities)

    predictions = np.array([[1.0, 2.0]], dtype=np.float64)
    with pytest.raises(ViolationError):
        voting_regressor_average(predictions, weights=(0.0, 0.0))
