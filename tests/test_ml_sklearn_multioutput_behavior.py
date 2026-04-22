from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyClassifier
from sklearn.multioutput import MultiOutputClassifier, RegressorChain


class MeanRegressor(RegressorMixin, BaseEstimator):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "MeanRegressor":
        del X
        self.constant_ = float(np.mean(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.constant_, dtype=np.float64)


def test_multioutput_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.multioutput import (
        chain_order_indices,
        chain_restore_output_order,
        chain_step_features,
        chain_training_features,
        multioutput_exact_match_score,
        multioutput_prediction_matrix,
    )

    assert callable(multioutput_prediction_matrix)
    assert callable(multioutput_exact_match_score)
    assert callable(chain_order_indices)
    assert callable(chain_training_features)
    assert callable(chain_step_features)
    assert callable(chain_restore_output_order)


def test_multioutput_prediction_matrix_matches_sklearn_transpose() -> None:
    from sciona.atoms.ml.sklearn.multioutput import multioutput_prediction_matrix

    output_predictions = np.array(
        [
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0],
            [5.0, 6.0, 7.0],
        ],
        dtype=np.float64,
    )
    assert np.array_equal(multioutput_prediction_matrix(output_predictions), np.asarray(output_predictions).T)


def test_multioutput_exact_match_score_matches_sklearn_score() -> None:
    from sciona.atoms.ml.sklearn.multioutput import multioutput_exact_match_score

    y_true = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    y_pred = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float64)

    classifier = MultiOutputClassifier(estimator=DummyClassifier())
    classifier.estimators_ = [DummyClassifier(), DummyClassifier()]
    classifier.predict = lambda X: y_pred

    assert multioutput_exact_match_score(y_true, y_pred) == classifier.score(np.zeros((4, 1)), y_true)
    assert multioutput_exact_match_score(y_true, y_pred) == 0.5


def test_chain_order_indices_match_regressor_chain_fit() -> None:
    from sciona.atoms.ml.sklearn.multioutput import chain_order_indices

    X = np.arange(24, dtype=np.float64).reshape(8, 3)
    Y = np.arange(32, dtype=np.float64).reshape(8, 4)

    assert np.array_equal(chain_order_indices(4), np.array([0, 1, 2, 3], dtype=np.int64))
    assert np.array_equal(chain_order_indices(4, order=(2, 0, 3, 1)), np.array([2, 0, 3, 1], dtype=np.int64))

    expected = RegressorChain(MeanRegressor(), order="random", random_state=13).fit(X, Y).order_
    assert np.array_equal(chain_order_indices(4, order="random", random_state=13), expected)


def test_chain_training_features_match_cv_none_hstack() -> None:
    from sciona.atoms.ml.sklearn.multioutput import chain_training_features

    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    Y = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]], dtype=np.float64)
    order = np.array([2, 0, 1], dtype=np.int64)

    expected = np.hstack((X, Y[:, order]))
    assert np.array_equal(chain_training_features(X, Y, order), expected)


def test_chain_step_features_and_restore_output_order_match_sklearn_logic() -> None:
    from sciona.atoms.ml.sklearn.multioutput import chain_restore_output_order, chain_step_features

    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    previous = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    assert np.array_equal(chain_step_features(X, previous), np.hstack((X, previous)))

    chain_predictions = np.array(
        [
            [20.0, 10.0, 30.0],
            [50.0, 40.0, 60.0],
        ],
        dtype=np.float64,
    )
    order = np.array([1, 0, 2], dtype=np.int64)
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(len(order))

    assert np.array_equal(chain_restore_output_order(chain_predictions, order), chain_predictions[:, inv_order])


def test_contracts_reject_invalid_order_and_shapes() -> None:
    from sciona.atoms.ml.sklearn.multioutput import chain_order_indices, chain_step_features

    with pytest.raises(ViolationError):
        chain_order_indices(3, order=(0, 0, 1))

    with pytest.raises(ViolationError):
        chain_step_features(
            np.array([[1.0, 2.0]], dtype=np.float64),
            np.array([[0.1], [0.2]], dtype=np.float64),
        )
