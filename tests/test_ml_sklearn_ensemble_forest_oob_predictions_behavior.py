from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sciona.atoms.ml.sklearn.ensemble.forest_sampling import (
    forest_generate_unsampled_indices,
)


def test_forest_oob_prediction_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_oob_predictions import (
        forest_classifier_oob_prediction_block,
        forest_oob_average_predictions,
        forest_oob_prediction_counts,
        forest_oob_prediction_totals,
        forest_oob_uncovered_mask,
        forest_regressor_oob_prediction_block,
    )

    assert callable(forest_classifier_oob_prediction_block)
    assert callable(forest_regressor_oob_prediction_block)
    assert callable(forest_oob_prediction_totals)
    assert callable(forest_oob_prediction_counts)
    assert callable(forest_oob_uncovered_mask)
    assert callable(forest_oob_average_predictions)


def test_forest_classifier_oob_prediction_block_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_oob_predictions import (
        forest_classifier_oob_prediction_block,
    )

    X, y = make_classification(
        n_samples=40,
        n_features=6,
        n_informative=5,
        n_redundant=0,
        random_state=7,
    )
    clf = RandomForestClassifier(
        n_estimators=3,
        bootstrap=True,
        oob_score=True,
        random_state=11,
    ).fit(X, y)
    tree = clf.estimators_[0]
    unsampled = forest_generate_unsampled_indices(
        clf._n_samples,
        clf._n_samples_bootstrap,
        random_state=int(tree.random_state),
    )
    X_unsampled = np.asarray(X[unsampled, :], dtype=np.float32)
    raw = np.asarray(tree.predict_proba(X_unsampled), dtype=np.float64)

    result = forest_classifier_oob_prediction_block(raw)
    expected = clf._get_oob_predictions(tree, X_unsampled)

    assert np.allclose(result, expected)


def test_forest_regressor_oob_prediction_block_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_oob_predictions import (
        forest_regressor_oob_prediction_block,
    )

    X, y = make_regression(n_samples=36, n_features=5, noise=0.2, random_state=13)
    reg = RandomForestRegressor(
        n_estimators=3,
        bootstrap=True,
        oob_score=True,
        random_state=17,
    ).fit(X, y)
    tree = reg.estimators_[0]
    unsampled = forest_generate_unsampled_indices(
        reg._n_samples,
        reg._n_samples_bootstrap,
        random_state=int(tree.random_state),
    )
    X_unsampled = np.asarray(X[unsampled, :], dtype=np.float32)
    raw = np.asarray(tree.predict(X_unsampled), dtype=np.float64)

    result = forest_regressor_oob_prediction_block(raw)
    expected = reg._get_oob_predictions(tree, X_unsampled)

    assert np.allclose(result, expected)


def test_forest_classifier_oob_totals_counts_mask_and_average_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_oob_predictions import (
        forest_classifier_oob_prediction_block,
        forest_oob_average_predictions,
        forest_oob_prediction_counts,
        forest_oob_prediction_totals,
        forest_oob_uncovered_mask,
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
    uncovered = forest_oob_uncovered_mask(counts)
    averaged = forest_oob_average_predictions(totals, counts)

    assert np.allclose(np.squeeze(averaged, axis=-1), clf.oob_decision_function_)
    assert np.array_equal(uncovered, np.all(counts == 0, axis=1))


def test_forest_regressor_oob_totals_counts_mask_and_average_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_oob_predictions import (
        forest_oob_average_predictions,
        forest_oob_prediction_counts,
        forest_oob_prediction_totals,
        forest_oob_uncovered_mask,
        forest_regressor_oob_prediction_block,
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
    uncovered = forest_oob_uncovered_mask(counts)
    averaged = forest_oob_average_predictions(totals, counts)

    assert np.allclose(np.squeeze(averaged, axis=(1, 2)), reg.oob_prediction_)
    assert np.array_equal(uncovered, np.all(counts == 0, axis=1))


def test_contracts_reject_invalid_forest_oob_prediction_inputs() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_oob_predictions import (
        forest_classifier_oob_prediction_block,
        forest_oob_average_predictions,
        forest_oob_prediction_counts,
        forest_oob_prediction_totals,
        forest_regressor_oob_prediction_block,
    )

    with pytest.raises(ViolationError):
        forest_classifier_oob_prediction_block(np.array([[0.2, 0.3]], dtype=np.float64))

    with pytest.raises(ViolationError):
        forest_regressor_oob_prediction_block(np.array([[[1.0]]], dtype=np.float64))

    with pytest.raises(ViolationError):
        forest_oob_prediction_counts(
            (np.array([0, 0], dtype=np.int64),),
            n_samples=4,
            n_outputs=1,
        )

    with pytest.raises(ViolationError):
        forest_oob_prediction_totals(
            (np.ones((2, 2, 1), dtype=np.float64),),
            (np.array([0], dtype=np.int64),),
            n_samples=4,
            prediction_width=2,
            n_outputs=1,
        )

    with pytest.raises(ViolationError):
        forest_oob_average_predictions(
            np.ones((3, 2, 1), dtype=np.float64),
            np.ones((2, 1), dtype=np.int64),
        )
