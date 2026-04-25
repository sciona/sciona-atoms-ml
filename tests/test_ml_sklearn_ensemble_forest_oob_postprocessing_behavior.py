from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sciona.atoms.ml.sklearn.ensemble.forest_oob_predictions import (
    forest_classifier_oob_prediction_block,
    forest_oob_average_predictions,
    forest_oob_prediction_counts,
    forest_oob_prediction_totals,
    forest_regressor_oob_prediction_block,
)
from sciona.atoms.ml.sklearn.ensemble.forest_sampling import (
    forest_generate_unsampled_indices,
)


def test_forest_oob_postprocessing_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_oob_postprocessing import (
        forest_classifier_oob_accuracy,
        forest_classifier_oob_decision_function,
        forest_regressor_oob_prediction,
        forest_regressor_oob_r2,
    )

    assert callable(forest_classifier_oob_decision_function)
    assert callable(forest_classifier_oob_accuracy)
    assert callable(forest_regressor_oob_prediction)
    assert callable(forest_regressor_oob_r2)


def test_forest_classifier_oob_postprocessing_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_oob_postprocessing import (
        forest_classifier_oob_accuracy,
        forest_classifier_oob_decision_function,
    )

    X, y = make_classification(
        n_samples=48,
        n_features=6,
        n_informative=5,
        n_redundant=0,
        random_state=19,
    )
    clf = RandomForestClassifier(
        n_estimators=5,
        bootstrap=True,
        oob_score=True,
        random_state=23,
    ).fit(X, y)

    unsampled_index_blocks = []
    prediction_blocks = []
    for tree in clf.estimators_:
        unsampled = forest_generate_unsampled_indices(
            clf._n_samples,
            clf._n_samples_bootstrap,
            random_state=int(tree.random_state),
        )
        unsampled_index_blocks.append(np.asarray(unsampled, dtype=np.int64))
        raw = np.asarray(tree.predict_proba(np.asarray(X[unsampled, :], dtype=np.float32)), dtype=np.float64)
        prediction_blocks.append(forest_classifier_oob_prediction_block(raw))

    totals = forest_oob_prediction_totals(
        tuple(prediction_blocks),
        tuple(unsampled_index_blocks),
        n_samples=clf._n_samples,
        prediction_width=int(clf.n_classes_),
        n_outputs=int(clf.n_outputs_),
    )
    counts = forest_oob_prediction_counts(
        tuple(unsampled_index_blocks),
        n_samples=clf._n_samples,
        n_outputs=int(clf.n_outputs_),
    )
    averaged = forest_oob_average_predictions(totals, counts)

    decision = forest_classifier_oob_decision_function(averaged)
    score = forest_classifier_oob_accuracy(np.asarray(y, dtype=np.int64), decision)

    assert np.allclose(decision, clf.oob_decision_function_)
    assert score == pytest.approx(float(clf.oob_score_))


def test_forest_classifier_oob_decision_function_preserves_multioutput_tensor() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_oob_postprocessing import (
        forest_classifier_oob_decision_function,
    )

    averaged = np.array(
        [
            [[0.7, 0.4], [0.3, 0.6]],
            [[0.2, 0.9], [0.8, 0.1]],
        ],
        dtype=np.float64,
    )

    result = forest_classifier_oob_decision_function(averaged)

    assert result.shape == averaged.shape
    assert np.allclose(result, averaged)


def test_forest_regressor_oob_postprocessing_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_oob_postprocessing import (
        forest_regressor_oob_prediction,
        forest_regressor_oob_r2,
    )

    X, y = make_regression(n_samples=42, n_features=5, noise=0.3, random_state=29)
    reg = RandomForestRegressor(
        n_estimators=4,
        bootstrap=True,
        oob_score=True,
        random_state=31,
    ).fit(X, y)

    unsampled_index_blocks = []
    prediction_blocks = []
    for tree in reg.estimators_:
        unsampled = forest_generate_unsampled_indices(
            reg._n_samples,
            reg._n_samples_bootstrap,
            random_state=int(tree.random_state),
        )
        unsampled_index_blocks.append(np.asarray(unsampled, dtype=np.int64))
        raw = np.asarray(tree.predict(np.asarray(X[unsampled, :], dtype=np.float32)), dtype=np.float64)
        prediction_blocks.append(forest_regressor_oob_prediction_block(raw))

    totals = forest_oob_prediction_totals(
        tuple(prediction_blocks),
        tuple(unsampled_index_blocks),
        n_samples=reg._n_samples,
        prediction_width=1,
        n_outputs=int(reg.n_outputs_),
    )
    counts = forest_oob_prediction_counts(
        tuple(unsampled_index_blocks),
        n_samples=reg._n_samples,
        n_outputs=int(reg.n_outputs_),
    )
    averaged = forest_oob_average_predictions(totals, counts)

    prediction = forest_regressor_oob_prediction(averaged)
    score = forest_regressor_oob_r2(np.asarray(y, dtype=np.float64), prediction)

    assert np.allclose(prediction, reg.oob_prediction_)
    assert score == pytest.approx(float(reg.oob_score_))


def test_forest_regressor_oob_prediction_preserves_multioutput_matrix() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_oob_postprocessing import (
        forest_regressor_oob_prediction,
    )

    averaged = np.array(
        [
            [[1.0, 2.0]],
            [[3.5, 4.5]],
        ],
        dtype=np.float64,
    )

    result = forest_regressor_oob_prediction(averaged)

    assert result.shape == (2, 2)
    assert np.allclose(result, np.array([[1.0, 2.0], [3.5, 4.5]], dtype=np.float64))


def test_contracts_reject_invalid_forest_oob_postprocessing_inputs() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_oob_postprocessing import (
        forest_classifier_oob_accuracy,
        forest_classifier_oob_decision_function,
        forest_regressor_oob_prediction,
        forest_regressor_oob_r2,
    )

    with pytest.raises(ViolationError):
        forest_classifier_oob_decision_function(np.array([[0.3, 0.7]], dtype=np.float64))

    with pytest.raises(ViolationError):
        forest_classifier_oob_accuracy(
            np.array([[0], [1]], dtype=np.int64),
            np.array([[0.4, 0.6], [0.8, 0.2]], dtype=np.float64),
        )

    with pytest.raises(ViolationError):
        forest_regressor_oob_prediction(np.ones((3, 2, 1), dtype=np.float64))

    with pytest.raises(ViolationError):
        forest_regressor_oob_r2(
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([[1.0], [2.0]], dtype=np.float64),
        )
