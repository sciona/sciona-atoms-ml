from __future__ import annotations

import numpy as np


def test_output_code_matrices_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.multiclass.output_code_matrices import (
        output_code_fit_classes,
        output_code_response_matrix,
        output_code_target_matrix,
    )

    assert callable(output_code_fit_classes)
    assert callable(output_code_target_matrix)
    assert callable(output_code_response_matrix)


def test_output_code_fit_classes_matches_sorted_unique() -> None:
    from sciona.atoms.ml.sklearn.multiclass.output_code_matrices import output_code_fit_classes

    y = np.array([3.0, 1.0, 3.0, 2.0, 1.0], dtype=np.float64)

    observed = output_code_fit_classes(y)

    assert np.array_equal(observed, np.array([1.0, 2.0, 3.0], dtype=np.float64))


def test_output_code_target_matrix_matches_sklearn_fit_gather() -> None:
    from sciona.atoms.ml.sklearn.multiclass.output_code_matrices import (
        output_code_fit_classes,
        output_code_target_matrix,
    )

    y = np.array([2.0, 0.0, 2.0, 1.0], dtype=np.float64)
    classes = output_code_fit_classes(y)
    code_book = np.array(
        [
            [1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0],
        ],
        dtype=np.float64,
    )

    classes_index = {class_value: index for index, class_value in enumerate(classes)}
    expected = np.array([code_book[classes_index[y[i]]] for i in range(y.shape[0])], dtype=int)

    observed = output_code_target_matrix(y, classes, code_book)

    assert np.array_equal(observed, expected)
    assert observed.dtype == np.int64


def test_output_code_response_matrix_matches_fortran_transpose_layout() -> None:
    from sciona.atoms.ml.sklearn.multiclass.output_code_matrices import output_code_response_matrix

    estimator_predictions = np.array(
        [
            [0.2, 0.5, 0.8],
            [-1.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    observed = output_code_response_matrix(estimator_predictions)
    expected = np.array(estimator_predictions, order="F", dtype=np.float64).T

    assert np.array_equal(observed, expected)
    assert observed.flags.c_contiguous
