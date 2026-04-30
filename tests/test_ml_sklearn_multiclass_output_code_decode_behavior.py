from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC

from sciona.atoms.ml.sklearn.multiclass.output_code_decode import (
    output_code_label_vector,
    output_code_nearest_class_indices,
    output_code_squared_distance_matrix,
)
from sciona.atoms.ml.sklearn.multiclass.output_code_matrices import output_code_response_matrix


def test_output_code_decode_atoms_import() -> None:
    assert callable(output_code_squared_distance_matrix)
    assert callable(output_code_nearest_class_indices)
    assert callable(output_code_label_vector)


def test_output_code_squared_distance_matrix_matches_manual_euclidean_distances() -> None:
    response_matrix = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    code_book = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float64)

    observed = output_code_squared_distance_matrix(response_matrix, code_book)
    expected = np.array([[0.0, 1.0, 5.0], [2.0, 1.0, 1.0]], dtype=np.float64)

    assert np.array_equal(observed, expected)


def test_output_code_nearest_class_indices_matches_argmin_tie_break() -> None:
    squared_distance_matrix = np.array([[2.0, 1.0, 1.0], [0.5, 0.7, 0.1]], dtype=np.float64)
    observed = output_code_nearest_class_indices(squared_distance_matrix)
    assert np.array_equal(observed, np.array([1, 2], dtype=np.int64))


def test_output_code_label_vector_matches_class_lookup() -> None:
    classes = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    class_indices = np.array([2, 0, 1], dtype=np.int64)
    observed = output_code_label_vector(classes, class_indices)
    assert np.array_equal(observed, np.array([30.0, 10.0, 20.0], dtype=np.float64))


def test_output_code_decode_atoms_match_sklearn_predict() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float64)
    y = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    clf = OutputCodeClassifier(LinearSVC(random_state=0), code_size=1.5, random_state=11)
    clf.fit(X, y)

    estimator_predictions = np.vstack([est.decision_function(X).ravel() for est in clf.estimators_])
    response_matrix = output_code_response_matrix(estimator_predictions)
    squared_distances = output_code_squared_distance_matrix(response_matrix, clf.code_book_)
    class_indices = output_code_nearest_class_indices(squared_distances)
    observed = output_code_label_vector(clf.classes_.astype(np.float64), class_indices)

    expected = clf.predict(X).astype(np.float64)
    assert np.array_equal(observed, expected)


def test_output_code_decode_atoms_reject_invalid_inputs() -> None:
    with pytest.raises(ViolationError):
        output_code_squared_distance_matrix(
            np.array([[0.0, 1.0]], dtype=np.float64),
            np.array([[0.0, 1.0, 2.0]], dtype=np.float64),
        )

    with pytest.raises(ViolationError):
        output_code_nearest_class_indices(np.array([[np.nan, 0.0]], dtype=np.float64))

    with pytest.raises(ViolationError):
        output_code_label_vector(
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([0, 2], dtype=np.int64),
        )
