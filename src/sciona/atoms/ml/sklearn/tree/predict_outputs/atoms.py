"""Sklearn tree prediction-output atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_tree_classifier_multioutput_labels,
    witness_tree_classifier_single_output_labels,
    witness_tree_regressor_multioutput_values,
    witness_tree_regressor_single_output_values,
)


def _finite_2d(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _finite_3d(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 3
        and array.shape[0] >= 1
        and array.shape[1] >= 1
        and array.shape[2] >= 1
        and np.all(np.isfinite(array))
    )


def _classes_block(values: object) -> bool:
    array = np.asarray(values, dtype=object)
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.unique(array).shape[0] == array.shape[0])


def _classes_blocks(values: object) -> bool:
    return bool(isinstance(values, tuple) and len(values) >= 1 and all(_classes_block(block) for block in values))


def _single_output_classifier_inputs(probabilities: object, classes: object) -> bool:
    if not (_finite_2d(probabilities) and _classes_block(classes)):
        return False
    matrix = np.asarray(probabilities, dtype=np.float64)
    class_values = np.asarray(classes, dtype=object)
    return bool(matrix.shape[1] == class_values.shape[0])


def _multioutput_classifier_inputs(probabilities: object, classes_blocks: object) -> bool:
    if not (_finite_3d(probabilities) and _classes_blocks(classes_blocks)):
        return False
    tensor = np.asarray(probabilities, dtype=np.float64)
    blocks = tuple(np.asarray(block, dtype=object) for block in classes_blocks)
    if tensor.shape[1] != len(blocks):
        return False
    return bool(all(tensor.shape[2] >= block.shape[0] >= 1 for block in blocks))


def _single_output_labels_valid(result: object, probabilities: object, classes: object) -> bool:
    if not _single_output_classifier_inputs(probabilities, classes):
        return False
    labels = np.asarray(result, dtype=object)
    matrix = np.asarray(probabilities, dtype=np.float64)
    class_values = np.asarray(classes, dtype=object)
    return bool(labels.shape == (matrix.shape[0],) and np.isin(labels, class_values).all())


def _multioutput_labels_valid(result: object, probabilities: object, classes_blocks: object) -> bool:
    if not _multioutput_classifier_inputs(probabilities, classes_blocks):
        return False
    labels = np.asarray(result, dtype=object)
    tensor = np.asarray(probabilities, dtype=np.float64)
    blocks = tuple(np.asarray(block, dtype=object) for block in classes_blocks)
    if labels.shape != (tensor.shape[0], tensor.shape[1]):
        return False
    return all(np.isin(labels[:, column], blocks[column]).all() for column in range(tensor.shape[1]))


def _single_output_regression_valid(result: object, probabilities: object) -> bool:
    if not _finite_2d(probabilities):
        return False
    values = np.asarray(result, dtype=np.float64)
    matrix = np.asarray(probabilities, dtype=np.float64)
    return bool(values.shape == (matrix.shape[0],) and np.allclose(values, matrix[:, 0]))


def _multioutput_regression_valid(result: object, probabilities: object) -> bool:
    if not _finite_3d(probabilities):
        return False
    values = np.asarray(result, dtype=np.float64)
    tensor = np.asarray(probabilities, dtype=np.float64)
    return bool(values.shape == (tensor.shape[0], tensor.shape[1]) and np.allclose(values, tensor[:, :, 0]))


@register_atom(witness_tree_classifier_single_output_labels)
@icontract.require(
    lambda probabilities, classes: _single_output_classifier_inputs(probabilities, classes),
    "probabilities and classes must align as a nonempty sample-by-class matrix and unique class vector",
)
@icontract.ensure(
    lambda result, probabilities, classes: _single_output_labels_valid(result, probabilities, classes),
    "single-output labels must match the sample count and come from classes",
)
def tree_classifier_single_output_labels(
    probabilities: NDArray[np.float64],
    classes: NDArray[np.object_],
) -> NDArray[np.object_]:
    """Decode single-output tree classifier labels from tree_.predict(X)."""
    matrix = np.asarray(probabilities, dtype=np.float64)
    class_values = np.asarray(classes, dtype=object)
    return np.asarray(class_values.take(np.argmax(matrix, axis=1), axis=0), dtype=object)


@register_atom(witness_tree_classifier_multioutput_labels)
@icontract.require(
    lambda probabilities, classes_blocks: _multioutput_classifier_inputs(probabilities, classes_blocks),
    "probabilities and classes_blocks must align as a nonempty tensor and class-block tuple",
)
@icontract.ensure(
    lambda result, probabilities, classes_blocks: _multioutput_labels_valid(result, probabilities, classes_blocks),
    "multioutput labels must match the sample count, output count, and per-output classes",
)
def tree_classifier_multioutput_labels(
    probabilities: NDArray[np.float64],
    classes_blocks: tuple[NDArray[np.object_], ...],
) -> NDArray[np.object_]:
    """Decode multioutput tree classifier labels from tree_.predict(X)."""
    tensor = np.asarray(probabilities, dtype=np.float64)
    predictions = np.empty((tensor.shape[0], tensor.shape[1]), dtype=object)
    for column, classes in enumerate(classes_blocks):
        class_values = np.asarray(classes, dtype=object)
        predictions[:, column] = class_values.take(np.argmax(tensor[:, column], axis=1), axis=0)
    return np.asarray(predictions, dtype=object)


@register_atom(witness_tree_regressor_single_output_values)
@icontract.require(lambda probabilities: _finite_2d(probabilities), "probabilities must be a nonempty finite 2D array")
@icontract.ensure(
    lambda result, probabilities: _single_output_regression_valid(result, probabilities),
    "single-output regression values must equal the first column of probabilities",
)
def tree_regressor_single_output_values(
    probabilities: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return single-output tree regressor values from tree_.predict(X)."""
    matrix = np.asarray(probabilities, dtype=np.float64)
    return np.asarray(matrix[:, 0], dtype=np.float64)


@register_atom(witness_tree_regressor_multioutput_values)
@icontract.require(lambda probabilities: _finite_3d(probabilities), "probabilities must be a nonempty finite 3D array")
@icontract.ensure(
    lambda result, probabilities: _multioutput_regression_valid(result, probabilities),
    "multioutput regression values must equal the first slice along the class axis",
)
def tree_regressor_multioutput_values(
    probabilities: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return multioutput tree regressor values from tree_.predict(X)."""
    tensor = np.asarray(probabilities, dtype=np.float64)
    return np.asarray(tensor[:, :, 0], dtype=np.float64)

