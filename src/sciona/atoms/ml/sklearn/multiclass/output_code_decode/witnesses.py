"""Ghost witnesses for output-code decode helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_output_code_squared_distance_matrix(
    response_matrix: AbstractArray,
    code_book: AbstractArray,
) -> AbstractArray:
    """Describe the sample-by-class squared Euclidean distance matrix used by output-code decoding."""
    if len(response_matrix.shape) != 2:
        raise ValueError("response_matrix must be a matrix")
    if len(code_book.shape) != 2:
        raise ValueError("code_book must be a matrix")
    if int(response_matrix.shape[1]) != int(code_book.shape[1]):
        raise ValueError("response_matrix and code_book must have matching estimator width")
    return AbstractArray(shape=(response_matrix.shape[0], code_book.shape[0]), dtype="float64")


def witness_output_code_nearest_class_indices(
    squared_distance_matrix: AbstractArray,
) -> AbstractArray:
    """Describe the nearest-code class indices selected from squared distances."""
    if len(squared_distance_matrix.shape) != 2:
        raise ValueError("squared_distance_matrix must be a matrix")
    return AbstractArray(shape=(squared_distance_matrix.shape[0],), dtype="int64")


def witness_output_code_label_vector(
    classes: AbstractArray,
    class_indices: AbstractArray,
) -> AbstractArray:
    """Describe the decoded output-code class label vector."""
    if len(classes.shape) != 1:
        raise ValueError("classes must be a vector")
    if len(class_indices.shape) != 1:
        raise ValueError("class_indices must be a vector")
    return AbstractArray(shape=class_indices.shape, dtype="float64")
