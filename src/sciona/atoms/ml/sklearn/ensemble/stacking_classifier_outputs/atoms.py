"""Stacking classifier output helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_stacking_classifier_labels_from_encoded,
    witness_stacking_classifier_multilabel_labels_from_encoded,
    witness_stacking_classifier_probability_matrix_from_blocks,
)

ObjectVector = NDArray[np.object_]
EncodedVector = NDArray[np.integer]
EncodedMatrix = NDArray[np.integer]
ProbabilityBlock = NDArray[np.float64]
ProbabilityBlockTuple = tuple[ProbabilityBlock, ...]
ClassBlockTuple = tuple[ObjectVector, ...]


def _class_vector_valid(classes: object) -> bool:
    values = np.asarray(classes, dtype=object)
    return bool(values.ndim == 1 and values.shape[0] >= 1)


def _encoded_vector_valid(encoded_labels: object, classes: object) -> bool:
    values = np.asarray(encoded_labels)
    class_values = np.asarray(classes, dtype=object)
    return bool(
        values.ndim == 1
        and values.shape[0] >= 1
        and np.issubdtype(values.dtype, np.integer)
        and _class_vector_valid(classes)
        and np.all(values >= 0)
        and np.all(values < class_values.shape[0])
    )


def _decoded_vector_valid(result: object, encoded_labels: object) -> bool:
    values = np.asarray(result, dtype=object)
    encoded = np.asarray(encoded_labels)
    return bool(values.shape == encoded.shape)


def _class_blocks_valid(classes_blocks: object) -> bool:
    return bool(
        isinstance(classes_blocks, tuple)
        and len(classes_blocks) >= 1
        and all(_class_vector_valid(block) for block in classes_blocks)
    )


def _encoded_matrix_valid(encoded_label_matrix: object, classes_blocks: object) -> bool:
    values = np.asarray(encoded_label_matrix)
    if not (
        values.ndim == 2
        and values.shape[0] >= 1
        and values.shape[1] >= 1
        and np.issubdtype(values.dtype, np.integer)
        and _class_blocks_valid(classes_blocks)
        and values.shape[1] == len(classes_blocks)
    ):
        return False
    for col_idx, classes in enumerate(classes_blocks):
        class_values = np.asarray(classes, dtype=object)
        column = values[:, col_idx]
        if not (np.all(column >= 0) and np.all(column < class_values.shape[0])):
            return False
    return True


def _decoded_matrix_valid(result: object, encoded_label_matrix: object) -> bool:
    values = np.asarray(result, dtype=object)
    encoded = np.asarray(encoded_label_matrix)
    return bool(values.shape == encoded.shape)


def _probability_blocks_valid(probability_blocks: object) -> bool:
    if not isinstance(probability_blocks, tuple) or len(probability_blocks) < 1:
        return False
    n_samples: int | None = None
    for block in probability_blocks:
        values = np.asarray(block, dtype=np.float64)
        if not (
            values.ndim == 2
            and values.shape[0] >= 1
            and values.shape[1] >= 2
            and np.all(np.isfinite(values))
            and np.all(values >= 0.0)
            and np.allclose(np.sum(values, axis=1), 1.0)
        ):
            return False
        if n_samples is None:
            n_samples = int(values.shape[0])
        elif int(values.shape[0]) != n_samples:
            return False
    return True


def _probability_matrix_valid(result: object, probability_blocks: object) -> bool:
    values = np.asarray(result, dtype=np.float64)
    if not isinstance(probability_blocks, tuple):
        return False
    return bool(
        values.ndim == 2
        and values.shape == (int(np.asarray(probability_blocks[0], dtype=np.float64).shape[0]), len(probability_blocks))
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and np.all(values <= 1.0)
    )


@register_atom(witness_stacking_classifier_labels_from_encoded)
@icontract.require(
    lambda encoded_labels, classes: _encoded_vector_valid(encoded_labels, classes),
    "encoded_labels must be a nonempty integer vector indexing the supplied class vector",
)
@icontract.ensure(
    lambda result, encoded_labels: _decoded_vector_valid(result, encoded_labels),
    "decoded labels must preserve the encoded vector shape",
)
def stacking_classifier_labels_from_encoded(
    encoded_labels: EncodedVector,
    classes: ObjectVector,
) -> NDArray[np.object_]:
    """Decode single-output stacking classifier predictions from encoded labels."""
    encoded = np.asarray(encoded_labels, dtype=np.int64)
    class_values = np.asarray(classes, dtype=object)
    return np.asarray(class_values.take(encoded, axis=0), dtype=object)


@register_atom(witness_stacking_classifier_multilabel_labels_from_encoded)
@icontract.require(
    lambda encoded_label_matrix, classes_blocks: _encoded_matrix_valid(encoded_label_matrix, classes_blocks),
    "encoded_label_matrix must be a nonempty integer matrix aligned with the class blocks",
)
@icontract.ensure(
    lambda result, encoded_label_matrix: _decoded_matrix_valid(result, encoded_label_matrix),
    "decoded multilabel outputs must preserve the encoded matrix shape",
)
def stacking_classifier_multilabel_labels_from_encoded(
    encoded_label_matrix: EncodedMatrix,
    classes_blocks: ClassBlockTuple,
) -> NDArray[np.object_]:
    """Decode multilabel stacking classifier predictions from one encoded column per output."""
    encoded = np.asarray(encoded_label_matrix, dtype=np.int64)
    decoded_columns = [
        np.asarray(np.asarray(classes, dtype=object).take(encoded[:, idx], axis=0), dtype=object)
        for idx, classes in enumerate(classes_blocks)
    ]
    return np.asarray(decoded_columns, dtype=object).T


@register_atom(witness_stacking_classifier_probability_matrix_from_blocks)
@icontract.require(
    lambda probability_blocks: _probability_blocks_valid(probability_blocks),
    "probability_blocks must be aligned nonempty normalized class-probability matrices",
)
@icontract.ensure(
    lambda result, probability_blocks: _probability_matrix_valid(result, probability_blocks),
    "probability matrix must have one column per block and preserve the shared sample count",
)
def stacking_classifier_probability_matrix_from_blocks(
    probability_blocks: ProbabilityBlockTuple,
) -> NDArray[np.float64]:
    """Convert per-output probability blocks into sklearn's multilabel stacking probability matrix."""
    return np.asarray([np.asarray(block, dtype=np.float64)[:, 0] for block in probability_blocks], dtype=np.float64).T
