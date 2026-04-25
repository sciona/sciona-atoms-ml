from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.ensemble import RandomForestClassifier


def test_forest_classifier_outputs_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_classifier_outputs import (
        forest_classifier_log_probability_blocks,
        forest_classifier_log_probability_matrix,
        forest_classifier_multioutput_labels,
    )

    assert callable(forest_classifier_log_probability_matrix)
    assert callable(forest_classifier_log_probability_blocks)
    assert callable(forest_classifier_multioutput_labels)


def test_forest_classifier_log_probability_matrix_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_classifier_outputs import (
        forest_classifier_log_probability_matrix,
    )

    X = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.2, 0.8], [0.8, 0.2]],
        dtype=np.float64,
    )
    y = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)
    clf = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)

    probabilities = np.asarray(clf.predict_proba(X), dtype=np.float64)
    result = forest_classifier_log_probability_matrix(probabilities)

    assert np.allclose(result, clf.predict_log_proba(X), equal_nan=False)


def test_forest_classifier_log_probability_blocks_matches_sklearn_multioutput() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_classifier_outputs import (
        forest_classifier_log_probability_blocks,
    )

    X = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.2, 0.8], [0.8, 0.2]],
        dtype=np.float64,
    )
    y = np.array([[0, 1], [0, 0], [1, 1], [1, 0], [0, 1], [1, 0]], dtype=np.int64)
    clf = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)

    probability_blocks = tuple(np.asarray(block, dtype=np.float64) for block in clf.predict_proba(X))
    result = forest_classifier_log_probability_blocks(probability_blocks)
    expected = clf.predict_log_proba(X)

    assert len(result) == len(expected)
    for actual, target in zip(result, expected):
        assert np.allclose(actual, target)


def test_forest_classifier_multioutput_labels_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_classifier_outputs import (
        forest_classifier_multioutput_labels,
    )

    X = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.2, 0.8], [0.8, 0.2]],
        dtype=np.float64,
    )
    y = np.array([["a", "up"], ["a", "down"], ["b", "up"], ["b", "down"], ["a", "up"], ["b", "down"]], dtype=object)
    clf = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)

    probability_blocks = tuple(np.asarray(block, dtype=np.float64) for block in clf.predict_proba(X))
    classes_blocks = tuple(np.asarray(block, dtype=object) for block in clf.classes_)
    result = forest_classifier_multioutput_labels(probability_blocks, classes_blocks)

    assert np.array_equal(result, clf.predict(X).astype(object))


def test_contracts_reject_invalid_forest_classifier_output_inputs() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_classifier_outputs import (
        forest_classifier_log_probability_blocks,
        forest_classifier_log_probability_matrix,
        forest_classifier_multioutput_labels,
    )

    with pytest.raises(ViolationError):
        forest_classifier_log_probability_matrix(np.array([[0.2, 0.2]], dtype=np.float64))

    with pytest.raises(ViolationError):
        forest_classifier_log_probability_blocks((np.array([[0.5, 0.5]], dtype=np.float64), np.array([[0.2, 0.2]], dtype=np.float64)))

    with pytest.raises(ViolationError):
        forest_classifier_multioutput_labels(
            (np.array([[0.5, 0.5]], dtype=np.float64),),
            (np.array(["a", "b", "c"], dtype=object),),
        )
