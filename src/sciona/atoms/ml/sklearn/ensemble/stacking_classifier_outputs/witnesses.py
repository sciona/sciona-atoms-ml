"""Ghost witnesses for sklearn stacking classifier output helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_stacking_classifier_labels_from_encoded(
    encoded_labels: AbstractArray,
    classes: AbstractArray,
) -> AbstractArray:
    """Describe single-output stacking classifier label decoding."""
    if len(encoded_labels.shape) != 1 or int(encoded_labels.shape[0]) < 1:
        raise ValueError("encoded_labels must be a nonempty 1D array")
    if len(classes.shape) != 1 or int(classes.shape[0]) < 1:
        raise ValueError("classes must be a nonempty 1D array")
    return AbstractArray(shape=(int(encoded_labels.shape[0]),), dtype="object")


def witness_stacking_classifier_multilabel_labels_from_encoded(
    encoded_label_matrix: AbstractArray,
    classes_blocks: tuple[AbstractArray, ...],
) -> AbstractArray:
    """Describe multilabel stacking classifier label decoding."""
    if len(encoded_label_matrix.shape) != 2 or int(encoded_label_matrix.shape[0]) < 1 or int(encoded_label_matrix.shape[1]) < 1:
        raise ValueError("encoded_label_matrix must be a nonempty 2D array")
    if len(classes_blocks) != int(encoded_label_matrix.shape[1]):
        raise ValueError("classes_blocks must match the encoded output width")
    for block in classes_blocks:
        if len(block.shape) != 1 or int(block.shape[0]) < 1:
            raise ValueError("each classes block must be a nonempty 1D array")
    return AbstractArray(
        shape=(int(encoded_label_matrix.shape[0]), int(encoded_label_matrix.shape[1])),
        dtype="object",
    )


def witness_stacking_classifier_probability_matrix_from_blocks(
    probability_blocks: tuple[AbstractArray, ...],
) -> AbstractArray:
    """Describe sklearn's multilabel stacking probability block conversion."""
    if len(probability_blocks) < 1:
        raise ValueError("probability_blocks must be nonempty")
    n_samples = int(probability_blocks[0].shape[0])
    for block in probability_blocks:
        if len(block.shape) != 2 or int(block.shape[0]) != n_samples or int(block.shape[1]) < 2:
            raise ValueError("probability blocks must be aligned 2D matrices with at least two columns")
    return AbstractArray(shape=(n_samples, len(probability_blocks)), dtype="float64")
