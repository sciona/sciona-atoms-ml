"""Output-code decode helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_output_code_label_vector,
    witness_output_code_nearest_class_indices,
    witness_output_code_squared_distance_matrix,
)


def _matrix_valid(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _decode_inputs_valid(response_matrix: object, code_book: object) -> bool:
    if not _matrix_valid(response_matrix) or not _matrix_valid(code_book):
        return False
    response_values = np.asarray(response_matrix, dtype=np.float64)
    code_values = np.asarray(code_book, dtype=np.float64)
    return bool(response_values.shape[1] == code_values.shape[1])


def _distance_matrix_valid(result: object, response_matrix: object, code_book: object) -> bool:
    if not _matrix_valid(result):
        return False
    values = np.asarray(result, dtype=np.float64)
    response_values = np.asarray(response_matrix, dtype=np.float64)
    code_values = np.asarray(code_book, dtype=np.float64)
    return bool(values.shape == (response_values.shape[0], code_values.shape[0]) and np.all(values >= 0.0))


def _class_vector_valid(classes: object) -> bool:
    try:
        array = np.asarray(classes, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 1
        and array.shape[0] >= 1
        and np.all(np.isfinite(array))
        and np.unique(array).shape[0] == array.shape[0]
    )


def _index_vector_valid(class_indices: object, n_classes: int) -> bool:
    try:
        array = np.asarray(class_indices, dtype=np.int64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all((0 <= array) & (array < n_classes)))


@register_atom(witness_output_code_squared_distance_matrix)
@icontract.require(
    lambda response_matrix, code_book: _decode_inputs_valid(response_matrix, code_book),
    "response_matrix and code_book must be finite matrices with matching estimator width",
)
@icontract.ensure(
    lambda result, response_matrix, code_book: _distance_matrix_valid(result, response_matrix, code_book),
    "result must be a nonnegative sample-by-class squared distance matrix",
)
def output_code_squared_distance_matrix(
    response_matrix: NDArray[np.float64],
    code_book: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute sample-by-class squared Euclidean distances to each output-code row."""
    response_values = np.asarray(response_matrix, dtype=np.float64)
    code_values = np.asarray(code_book, dtype=np.float64)
    differences = response_values[:, np.newaxis, :] - code_values[np.newaxis, :, :]
    return np.sum(differences * differences, axis=2, dtype=np.float64)


@register_atom(witness_output_code_nearest_class_indices)
@icontract.require(lambda squared_distance_matrix: _matrix_valid(squared_distance_matrix), "squared_distance_matrix must be a finite matrix")
@icontract.ensure(
    lambda result, squared_distance_matrix: _index_vector_valid(result, np.asarray(squared_distance_matrix).shape[1]),
    "result must be a valid nearest-class index vector",
)
def output_code_nearest_class_indices(
    squared_distance_matrix: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Select sklearn's nearest-code class index for each sample by rowwise argmin."""
    return np.asarray(np.argmin(np.asarray(squared_distance_matrix, dtype=np.float64), axis=1), dtype=np.int64)


@register_atom(witness_output_code_label_vector)
@icontract.require(lambda classes: _class_vector_valid(classes), "classes must be a finite unique class vector")
@icontract.require(
    lambda classes, class_indices: _index_vector_valid(class_indices, np.asarray(classes, dtype=np.float64).shape[0]),
    "class_indices must be valid positions into classes",
)
@icontract.ensure(
    lambda result, class_indices: np.asarray(result, dtype=np.float64).shape == np.asarray(class_indices, dtype=np.int64).shape,
    "result must preserve the class-index vector shape",
)
def output_code_label_vector(
    classes: NDArray[np.float64],
    class_indices: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Decode nearest-code indices into sklearn's output-code class labels."""
    class_values = np.asarray(classes, dtype=np.float64)
    index_values = np.asarray(class_indices, dtype=np.int64)
    return np.asarray(class_values[index_values], dtype=np.float64)
