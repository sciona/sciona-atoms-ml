from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


def test_bagging_classifier_io_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_classifier_io import (
        bagging_classifier_fit_targets,
        bagging_classifier_labels_from_probabilities,
    )

    assert callable(bagging_classifier_fit_targets)
    assert callable(bagging_classifier_labels_from_probabilities)


def test_bagging_classifier_fit_targets_matches_private_validate_y() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_classifier_io import bagging_classifier_fit_targets

    y = np.array(["beta", "alpha", "beta", "gamma", "alpha"], dtype=object)
    clf = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=1, random_state=0), n_estimators=3, random_state=0)

    encoded = clf._validate_y(y)
    state, actual_encoded = bagging_classifier_fit_targets(y)

    assert np.array_equal(state.classes, clf.classes_.astype(object))
    assert state.n_classes == int(clf.n_classes_)
    assert np.array_equal(actual_encoded, encoded.astype(np.int64))


def test_bagging_classifier_labels_from_probabilities_matches_predict() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_classifier_io import bagging_classifier_labels_from_probabilities
    from sciona.atoms.ml.sklearn.ensemble.bagging_aggregation import bagging_classifier_average_probabilities

    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.2, 0.8],
            [0.8, 0.2],
        ],
        dtype=np.float64,
    )
    y = np.array(["no", "no", "yes", "yes", "no", "yes"], dtype=object)
    clf = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=2, random_state=0),
        n_estimators=5,
        random_state=3,
    ).fit(X, y)

    probability_blocks = np.stack(
        [estimator.predict_proba(X[:, features]) for estimator, features in zip(clf.estimators_, clf.estimators_features_)],
        axis=0,
    )
    class_index_blocks = tuple(
        np.asarray(estimator.classes_, dtype=np.int64)
        for estimator in clf.estimators_
    )
    averaged = bagging_classifier_average_probabilities(
        tuple(np.asarray(block, dtype=np.float64) for block in probability_blocks),
        class_index_blocks,
        n_classes=int(clf.classes_.shape[0]),
    )
    result = bagging_classifier_labels_from_probabilities(averaged, clf.classes_)

    assert np.array_equal(result, clf.predict(X).astype(object))


def test_contracts_reject_invalid_bagging_classifier_io_inputs() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_classifier_io import (
        bagging_classifier_fit_targets,
        bagging_classifier_labels_from_probabilities,
    )

    with pytest.raises(ViolationError):
        bagging_classifier_fit_targets(np.empty((0,), dtype=np.int64))

    with pytest.raises(ViolationError):
        bagging_classifier_fit_targets(np.array([[0, 1], [1, 0]], dtype=np.int64))

    with pytest.raises(ViolationError):
        bagging_classifier_labels_from_probabilities(
            np.array([[0.2, 0.2]], dtype=np.float64),
            np.array(["a", "b"], dtype=object),
        )
