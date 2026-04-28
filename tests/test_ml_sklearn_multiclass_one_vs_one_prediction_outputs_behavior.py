from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import _threshold_for_binary_predict

from sciona.atoms.ml.sklearn.multiclass.one_vs_one_prediction_outputs import (
    one_vs_one_binary_labels,
    one_vs_one_multiclass_labels,
)


class BinaryDecisionClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BinaryDecisionClassifier":
        self.classes_ = np.unique(y)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        del X
        return np.array([0.0], dtype=np.float64)


def test_one_vs_one_prediction_output_atoms_import() -> None:
    assert callable(one_vs_one_binary_labels)
    assert callable(one_vs_one_multiclass_labels)


def test_one_vs_one_binary_labels_match_sklearn_predict_rule() -> None:
    scores = np.array([-0.2, 0.0, 0.4], dtype=np.float64)
    classes = np.array([10.0, 20.0], dtype=np.float64)
    threshold = _threshold_for_binary_predict(BinaryDecisionClassifier())

    observed = one_vs_one_binary_labels(scores, classes, threshold=threshold)

    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([10.0, 20.0, 10.0, 20.0], dtype=np.float64)
    ovo = OneVsOneClassifier(BinaryDecisionClassifier()).fit(X, y)
    ovo.decision_function = lambda X: scores

    expected = ovo.predict(np.zeros((scores.shape[0], 1), dtype=np.float64))
    assert np.array_equal(observed, expected)


def test_one_vs_one_multiclass_labels_match_argmax_predict_rule() -> None:
    scores = np.array(
        [
            [0.1, 0.7, 0.2],
            [1.0, 0.4, 0.6],
            [0.5, 0.5, 0.2],
        ],
        dtype=np.float64,
    )
    classes = np.array([10.0, 20.0, 30.0], dtype=np.float64)

    observed = one_vs_one_multiclass_labels(scores, classes)

    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float64)
    y = np.array([10.0, 20.0, 30.0, 10.0, 20.0, 30.0], dtype=np.float64)
    ovo = OneVsOneClassifier(BinaryDecisionClassifier()).fit(X, y)
    ovo.decision_function = lambda X: scores

    expected = ovo.predict(np.zeros((scores.shape[0], 1), dtype=np.float64))
    assert np.array_equal(observed, expected)


def test_one_vs_one_binary_labels_respect_threshold_tie_break() -> None:
    scores = np.array([0.0, 0.1], dtype=np.float64)
    classes = np.array([1.0, 2.0], dtype=np.float64)

    observed = one_vs_one_binary_labels(scores, classes, threshold=0.0)

    assert np.array_equal(observed, np.array([1.0, 2.0], dtype=np.float64))


def test_one_vs_one_prediction_output_contracts_reject_invalid_shapes() -> None:
    with pytest.raises(ViolationError):
        one_vs_one_binary_labels(
            np.array([[0.1, 0.2]], dtype=np.float64),
            np.array([1.0, 2.0], dtype=np.float64),
        )

    with pytest.raises(ViolationError):
        one_vs_one_multiclass_labels(
            np.array([0.1, 0.2], dtype=np.float64),
            np.array([1.0, 2.0], dtype=np.float64),
        )
