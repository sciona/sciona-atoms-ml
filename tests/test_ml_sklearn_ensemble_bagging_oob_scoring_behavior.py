from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def _oob_rows(sample_indices: np.ndarray, n_samples: int) -> np.ndarray:
    in_bag = np.zeros(n_samples, dtype=bool)
    in_bag[np.asarray(sample_indices, dtype=np.int64)] = True
    return np.flatnonzero(~in_bag)


def test_bagging_oob_scoring_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_oob_scoring import (
        bagging_classifier_oob_accuracy,
        bagging_oob_uncovered_mask,
        bagging_regressor_oob_r2,
    )

    assert callable(bagging_oob_uncovered_mask)
    assert callable(bagging_classifier_oob_accuracy)
    assert callable(bagging_regressor_oob_r2)


def test_bagging_oob_uncovered_mask_marks_never_held_out_samples() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_oob_scoring import bagging_oob_uncovered_mask

    sample_index_blocks = (
        np.array([0, 1, 2], dtype=np.int64),
        np.array([0, 2, 3], dtype=np.int64),
    )
    result = bagging_oob_uncovered_mask(sample_index_blocks, n_samples=4)
    expected = np.array([True, False, True, False], dtype=np.bool_)
    assert np.array_equal(result, expected)


def test_bagging_classifier_oob_accuracy_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_classifier_io import bagging_classifier_fit_targets
    from sciona.atoms.ml.sklearn.ensemble.bagging_oob import bagging_classifier_oob_probability_totals
    from sciona.atoms.ml.sklearn.ensemble.bagging_oob_scoring import (
        bagging_classifier_oob_accuracy,
        bagging_oob_uncovered_mask,
    )

    X, y = make_classification(
        n_samples=120,
        n_features=6,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        weights=[0.7, 0.2, 0.1],
        random_state=7,
    )
    clf = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=0),
        n_estimators=25,
        bootstrap=True,
        oob_score=True,
        random_state=11,
    ).fit(X, y)

    sample_index_blocks = tuple(np.asarray(samples, dtype=np.int64) for samples in clf.estimators_samples_)
    probability_blocks = []
    class_index_blocks = []
    for estimator, sample_indices, features in zip(clf.estimators_, sample_index_blocks, clf.estimators_features_):
        rows = _oob_rows(sample_indices, X.shape[0])
        probability_blocks.append(np.asarray(estimator.predict_proba(X[rows][:, features]), dtype=np.float64))
        class_index_blocks.append(np.searchsorted(clf.classes_, estimator.classes_).astype(np.int64))

    totals = bagging_classifier_oob_probability_totals(
        tuple(probability_blocks),
        tuple(class_index_blocks),
        sample_index_blocks,
        n_samples=X.shape[0],
        n_classes=int(clf.classes_.shape[0]),
    )
    _, encoded = bagging_classifier_fit_targets(y.astype(np.int64))
    accuracy = bagging_classifier_oob_accuracy(encoded, totals)
    uncovered = bagging_oob_uncovered_mask(sample_index_blocks, n_samples=X.shape[0])

    assert np.isclose(accuracy, clf.oob_score_)
    assert np.array_equal(uncovered, np.sum(totals, axis=1) == 0.0)


def test_bagging_regressor_oob_r2_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_oob import bagging_regressor_oob_predictions
    from sciona.atoms.ml.sklearn.ensemble.bagging_oob_scoring import (
        bagging_oob_uncovered_mask,
        bagging_regressor_oob_r2,
    )

    X, y = make_regression(n_samples=110, n_features=5, noise=0.4, random_state=19)
    reg = BaggingRegressor(
        estimator=DecisionTreeRegressor(random_state=0),
        n_estimators=20,
        bootstrap=True,
        oob_score=True,
        random_state=23,
    )
    reg.fit(X, y)

    sample_index_blocks = tuple(np.asarray(samples, dtype=np.int64) for samples in reg.estimators_samples_)
    prediction_blocks = tuple(
        np.asarray(estimator.predict(X[_oob_rows(sample_indices, X.shape[0])][:, features]), dtype=np.float64)
        for estimator, sample_indices, features in zip(reg.estimators_, sample_index_blocks, reg.estimators_features_)
    )
    predictions = bagging_regressor_oob_predictions(
        prediction_blocks,
        sample_index_blocks,
        n_samples=X.shape[0],
    )
    r2_value = bagging_regressor_oob_r2(y, predictions)
    uncovered = bagging_oob_uncovered_mask(sample_index_blocks, n_samples=X.shape[0])

    assert np.isclose(r2_value, reg.oob_score_)
    assert np.array_equal(uncovered, np.array([(sample_count == 0) for sample_count in np.bincount(np.concatenate([_oob_rows(b, X.shape[0]) for b in sample_index_blocks]), minlength=X.shape[0])], dtype=np.bool_))


def test_contracts_reject_invalid_bagging_oob_scoring_inputs() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_oob_scoring import (
        bagging_classifier_oob_accuracy,
        bagging_oob_uncovered_mask,
        bagging_regressor_oob_r2,
    )

    with pytest.raises(ViolationError):
        bagging_oob_uncovered_mask((np.array([0, 4], dtype=np.int64),), n_samples=4)

    with pytest.raises(ViolationError):
        bagging_classifier_oob_accuracy(
            np.array([0, 2], dtype=np.int64),
            np.array([[0.2, 0.8], [0.7, 0.3]], dtype=np.float64),
        )

    with pytest.raises(ViolationError):
        bagging_regressor_oob_r2(
            np.array([1.0, np.nan], dtype=np.float64),
            np.array([1.0, 2.0], dtype=np.float64),
        )
