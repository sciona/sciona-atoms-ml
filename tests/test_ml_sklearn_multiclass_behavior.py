from __future__ import annotations

import itertools

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import pairwise_distances_argmin
from sklearn.multiclass import OutputCodeClassifier
from sklearn.utils.multiclass import _ovr_decision_function


class DecisionScoreClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionScoreClassifier":
        self.classes_ = np.unique(y)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0], dtype=np.float64)


class ProbabilityClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ProbabilityClassifier":
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.tile(np.array([[0.4, 0.6]], dtype=np.float64), (X.shape[0], 1))


class ColumnScoreClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, values: tuple[float, ...]) -> None:
        self.values = values

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        del X
        return np.asarray(self.values, dtype=np.float64)


def test_multiclass_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.multiclass import (
        one_vs_one_class_pairs,
        one_vs_one_decision_scores,
        one_vs_rest_binary_indicator,
        one_vs_rest_multiclass_labels,
        output_code_book,
        output_code_decode,
    )

    assert callable(one_vs_rest_multiclass_labels)
    assert callable(one_vs_rest_binary_indicator)
    assert callable(one_vs_one_decision_scores)
    assert callable(one_vs_one_class_pairs)
    assert callable(output_code_book)
    assert callable(output_code_decode)


def test_one_vs_rest_multiclass_labels_use_argmax_tie_rule() -> None:
    from sciona.atoms.ml.sklearn.multiclass import one_vs_rest_multiclass_labels

    responses = np.array(
        [
            [0.1, 0.6, 0.3],
            [2.0, 2.0, 1.0],
            [-1.0, -0.5, -0.25],
        ],
        dtype=np.float64,
    )
    classes = np.array([10.0, 20.0, 30.0], dtype=np.float64)

    result = one_vs_rest_multiclass_labels(responses, classes)
    assert np.array_equal(result, classes[np.argmax(responses, axis=1)])
    assert np.array_equal(result, np.array([20.0, 10.0, 30.0]))


def test_one_vs_rest_binary_indicator_thresholds_responses() -> None:
    from sciona.atoms.ml.sklearn.multiclass import one_vs_rest_binary_indicator

    responses = np.array([[0.1, 0.6, -0.2], [0.5, 0.2, 1.1]], dtype=np.float64)
    assert np.array_equal(
        one_vs_rest_binary_indicator(responses, threshold=0.5),
        np.array([[False, True, False], [False, False, True]], dtype=np.bool_),
    )


def test_one_vs_one_decision_scores_match_sklearn_helper() -> None:
    from sciona.atoms.ml.sklearn.multiclass import one_vs_one_decision_scores

    predictions = np.array(
        [
            [0, 1, 1, 0, 1, 0],
            [1, 1, 0, 0, 0, 1],
        ],
        dtype=np.int64,
    )
    confidences = np.array(
        [
            [0.8, -0.2, 1.4, 0.7, -1.2, 0.1],
            [-0.5, 0.3, 0.9, -0.4, 0.2, 1.3],
        ],
        dtype=np.float64,
    )

    result = one_vs_one_decision_scores(predictions, confidences, n_classes=4)
    expected = _ovr_decision_function(predictions, confidences, 4)
    assert np.allclose(result, expected)


def test_one_vs_one_class_pairs_match_sklearn_loop_order() -> None:
    from sciona.atoms.ml.sklearn.multiclass import one_vs_one_class_pairs

    result = one_vs_one_class_pairs(5)
    expected = np.asarray(list(itertools.combinations(range(5), 2)), dtype=np.int64)
    assert np.array_equal(result, expected)


def test_output_code_book_matches_sklearn_for_decision_and_probability_estimators() -> None:
    from sciona.atoms.ml.sklearn.multiclass import output_code_book

    X = np.arange(24, dtype=np.float64).reshape(8, 3)
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)

    decision_expected = OutputCodeClassifier(
        DecisionScoreClassifier(),
        code_size=1.25,
        random_state=17,
    ).fit(X, y)
    assert np.array_equal(
        output_code_book(4, code_size=1.25, random_state=17, estimator_has_decision_function=True),
        decision_expected.code_book_,
    )

    probability_expected = OutputCodeClassifier(
        ProbabilityClassifier(),
        code_size=1.25,
        random_state=17,
    ).fit(X, y)
    assert np.array_equal(
        output_code_book(4, code_size=1.25, random_state=17, estimator_has_decision_function=False),
        probability_expected.code_book_,
    )


def test_output_code_decode_matches_sklearn_nearest_code_row() -> None:
    from sciona.atoms.ml.sklearn.multiclass import output_code_decode

    code_book = np.array(
        [
            [1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0],
        ],
        dtype=np.float64,
    )
    responses = np.array(
        [
            [0.9, -0.7, 0.8],
            [-0.5, 0.9, -0.8],
            [0.4, 0.8, -0.6],
        ],
        dtype=np.float64,
    )
    classes = np.array([2.0, 4.0, 8.0], dtype=np.float64)

    expected = classes[pairwise_distances_argmin(responses, code_book, metric="euclidean")]
    assert np.array_equal(output_code_decode(responses, code_book, classes), expected)

    estimator = OutputCodeClassifier(DecisionScoreClassifier())
    estimator.classes_ = classes
    estimator.code_book_ = code_book
    estimator.estimators_ = [ColumnScoreClassifier(tuple(responses[:, i])) for i in range(responses.shape[1])]
    assert np.array_equal(estimator.predict(np.zeros((responses.shape[0], 1))), expected)


def test_contracts_reject_invalid_dimensions() -> None:
    from sciona.atoms.ml.sklearn.multiclass import (
        one_vs_one_decision_scores,
        output_code_decode,
    )

    with pytest.raises(ViolationError):
        one_vs_one_decision_scores(
            np.array([[0, 1]], dtype=np.int64),
            np.array([[0.1, 0.2]], dtype=np.float64),
            n_classes=4,
        )

    with pytest.raises(ViolationError):
        output_code_decode(
            np.array([[0.1, 0.2]], dtype=np.float64),
            np.array([[1.0, -1.0, 1.0]], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
        )
