from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.multiclass import OneVsRestClassifier

from sciona.atoms.ml.sklearn.multiclass.one_vs_rest_prediction_outputs import (
    one_vs_rest_predict_argmaxima_init,
    one_vs_rest_predict_labels_from_argmaxima,
    one_vs_rest_predict_maxima_init,
    one_vs_rest_predict_multiclass_update,
)


class BinaryScoreClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BinaryScoreClassifier":
        self.classes_ = np.unique(y)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0], dtype=np.float64)


def test_one_vs_rest_prediction_output_atoms_import() -> None:
    assert callable(one_vs_rest_predict_maxima_init)
    assert callable(one_vs_rest_predict_argmaxima_init)
    assert callable(one_vs_rest_predict_multiclass_update)
    assert callable(one_vs_rest_predict_labels_from_argmaxima)


def test_one_vs_rest_predict_init_helpers_match_sklearn_shapes() -> None:
    maxima = one_vs_rest_predict_maxima_init(3)
    argmaxima = one_vs_rest_predict_argmaxima_init(3)

    assert np.array_equal(maxima, np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64))
    assert np.array_equal(argmaxima, np.array([0, 0, 0], dtype=np.int64))


def test_one_vs_rest_predict_update_matches_later_tie_wins_rule() -> None:
    maxima = one_vs_rest_predict_maxima_init(3)
    argmaxima = one_vs_rest_predict_argmaxima_init(3)

    maxima, argmaxima = one_vs_rest_predict_multiclass_update(
        maxima,
        argmaxima,
        np.array([2.0, 0.0, 3.0], dtype=np.float64),
        class_index=0,
    )
    maxima, argmaxima = one_vs_rest_predict_multiclass_update(
        maxima,
        argmaxima,
        np.array([2.0, 1.0, 3.0], dtype=np.float64),
        class_index=1,
    )
    maxima, argmaxima = one_vs_rest_predict_multiclass_update(
        maxima,
        argmaxima,
        np.array([1.0, 1.0, 3.0], dtype=np.float64),
        class_index=2,
    )

    assert np.array_equal(maxima, np.array([2.0, 1.0, 3.0], dtype=np.float64))
    assert np.array_equal(argmaxima, np.array([1, 2, 2], dtype=np.int64))


def test_one_vs_rest_predict_labels_from_argmaxima_decodes_classes() -> None:
    argmaxima = np.array([1, 2, 0], dtype=np.int64)
    classes = np.array([10.0, 20.0, 30.0], dtype=np.float64)

    observed = one_vs_rest_predict_labels_from_argmaxima(argmaxima, classes)

    assert np.array_equal(observed, np.array([20.0, 30.0, 10.0], dtype=np.float64))


def test_one_vs_rest_prediction_outputs_match_sklearn_multiclass_predict() -> None:
    responses = [
        np.array([2.0, 0.0, 3.0], dtype=np.float64),
        np.array([2.0, 1.0, 3.0], dtype=np.float64),
        np.array([1.0, 1.0, 3.0], dtype=np.float64),
    ]
    classes = np.array([10.0, 20.0, 30.0], dtype=np.float64)

    maxima = one_vs_rest_predict_maxima_init(3)
    argmaxima = one_vs_rest_predict_argmaxima_init(3)
    for class_index, pred in enumerate(responses):
        maxima, argmaxima = one_vs_rest_predict_multiclass_update(
            maxima,
            argmaxima,
            pred,
            class_index=class_index,
        )
    observed = one_vs_rest_predict_labels_from_argmaxima(argmaxima, classes)

    X = np.arange(18, dtype=np.float64).reshape(6, 3)
    y = np.array([10.0, 20.0, 30.0, 10.0, 20.0, 30.0], dtype=np.float64)
    ovr = OneVsRestClassifier(BinaryScoreClassifier()).fit(X, y)
    ovr.classes_ = classes
    ovr.label_binarizer_.classes_ = classes
    ovr.label_binarizer_.y_type_ = "multiclass"
    ovr.estimators_ = [BinaryScoreClassifier(), BinaryScoreClassifier(), BinaryScoreClassifier()]
    response_iter = iter(responses)
    ovr._predict_binary = None
    import sklearn.multiclass as multiclass_module
    original_predict_binary = multiclass_module._predict_binary
    multiclass_module._predict_binary = lambda e, X: next(response_iter)
    try:
        expected = ovr.predict(np.zeros((3, 1), dtype=np.float64))
    finally:
        multiclass_module._predict_binary = original_predict_binary

    assert np.array_equal(observed, expected)


def test_one_vs_rest_prediction_output_contracts_reject_invalid_shapes() -> None:
    with pytest.raises(ViolationError):
        one_vs_rest_predict_multiclass_update(
            np.array([-np.inf, -np.inf], dtype=np.float64),
            np.array([0, 0, 0], dtype=np.int64),
            np.array([1.0, 2.0], dtype=np.float64),
            class_index=0,
        )

    with pytest.raises(ViolationError):
        one_vs_rest_predict_labels_from_argmaxima(
            np.array([0, 2], dtype=np.int64),
            np.array([10.0, 20.0], dtype=np.float64),
        )
