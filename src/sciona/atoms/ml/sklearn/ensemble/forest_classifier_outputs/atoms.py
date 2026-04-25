"""Functions for forest classifier output postprocessing."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_forest_classifier_log_probability_blocks,
    witness_forest_classifier_log_probability_matrix,
    witness_forest_classifier_multioutput_labels,
)

ProbabilityBlock = NDArray[np.float64]
ProbabilityBlockTuple = tuple[ProbabilityBlock, ...]
ClassBlockTuple = tuple[NDArray[np.object_], ...]


def _probability_matrix_valid(probabilities: object) -> bool:
    try:
        values = np.asarray(probabilities, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        values.ndim == 2
        and values.shape[0] >= 1
        and values.shape[1] >= 1
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and np.allclose(np.sum(values, axis=1), 1.0)
    )


def _log_probability_matrix_valid(result: object, probabilities: object) -> bool:
    values = np.asarray(result, dtype=np.float64)
    input_values = np.asarray(probabilities, dtype=np.float64)
    return bool(values.shape == input_values.shape and np.all(np.isnan(values) == np.isnan(values)) and np.all(np.isfinite(values) | np.isneginf(values)))


def _probability_blocks_valid(probability_blocks: object) -> bool:
    return bool(
        isinstance(probability_blocks, tuple)
        and len(probability_blocks) >= 1
        and all(_probability_matrix_valid(block) for block in probability_blocks)
    )


def _log_probability_blocks_valid(result: object, probability_blocks: object) -> bool:
    if not isinstance(result, tuple) or not isinstance(probability_blocks, tuple):
        return False
    if len(result) != len(probability_blocks):
        return False
    return all(_log_probability_matrix_valid(block_result, block_input) for block_result, block_input in zip(result, probability_blocks))


def _classes_block_valid(classes: object) -> bool:
    values = np.asarray(classes, dtype=object)
    return bool(values.ndim == 1 and values.shape[0] >= 1 and np.unique(values).shape[0] == values.shape[0])


def _aligned_multioutput_inputs(probability_blocks: object, classes_blocks: object) -> bool:
    if not (
        isinstance(probability_blocks, tuple)
        and isinstance(classes_blocks, tuple)
        and len(probability_blocks) >= 1
        and len(probability_blocks) == len(classes_blocks)
    ):
        return False
    n_samples = None
    for probabilities, classes in zip(probability_blocks, classes_blocks):
        if not (_probability_matrix_valid(probabilities) and _classes_block_valid(classes)):
            return False
        matrix = np.asarray(probabilities, dtype=np.float64)
        class_values = np.asarray(classes, dtype=object)
        if matrix.shape[1] != class_values.shape[0]:
            return False
        if n_samples is None:
            n_samples = int(matrix.shape[0])
        elif int(matrix.shape[0]) != n_samples:
            return False
    return True


def _multioutput_labels_valid(result: object, probability_blocks: object, classes_blocks: object) -> bool:
    if not isinstance(probability_blocks, tuple) or not isinstance(classes_blocks, tuple):
        return False
    values = np.asarray(result, dtype=object)
    n_samples = int(np.asarray(probability_blocks[0], dtype=np.float64).shape[0])
    if values.shape != (n_samples, len(probability_blocks)):
        return False
    for column, classes in enumerate(classes_blocks):
        class_values = np.asarray(classes, dtype=object)
        if not np.isin(values[:, column], class_values).all():
            return False
    return True


@register_atom(witness_forest_classifier_log_probability_matrix)
@icontract.require(
    lambda probabilities: _probability_matrix_valid(probabilities),
    "probabilities must be a normalized nonnegative sample-by-class matrix",
)
@icontract.ensure(
    lambda result, probabilities: _log_probability_matrix_valid(result, probabilities),
    "log probability matrix must preserve shape and contain finite or -inf entries",
)
def forest_classifier_log_probability_matrix(
    probabilities: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Take sklearn's single-output forest class probabilities into log space."""
    return np.asarray(np.log(np.asarray(probabilities, dtype=np.float64)), dtype=np.float64)


@register_atom(witness_forest_classifier_log_probability_blocks)
@icontract.require(
    lambda probability_blocks: _probability_blocks_valid(probability_blocks),
    "probability_blocks must be a nonempty tuple of normalized nonnegative matrices",
)
@icontract.ensure(
    lambda result, probability_blocks: _log_probability_blocks_valid(result, probability_blocks),
    "log probability blocks must preserve the tuple length and each block shape",
)
def forest_classifier_log_probability_blocks(
    probability_blocks: ProbabilityBlockTuple,
) -> ProbabilityBlockTuple:
    """Take sklearn's multioutput forest class probability blocks into log space."""
    return tuple(
        np.asarray(np.log(np.asarray(block, dtype=np.float64)), dtype=np.float64)
        for block in probability_blocks
    )


@register_atom(witness_forest_classifier_multioutput_labels)
@icontract.require(
    lambda probability_blocks, classes_blocks: _aligned_multioutput_inputs(probability_blocks, classes_blocks),
    "probability_blocks and classes_blocks must be nonempty aligned tuples with matching class widths",
)
@icontract.ensure(
    lambda result, probability_blocks, classes_blocks: _multioutput_labels_valid(result, probability_blocks, classes_blocks),
    "multioutput labels must match the sample count and come from each output's classes block",
)
def forest_classifier_multioutput_labels(
    probability_blocks: ProbabilityBlockTuple,
    classes_blocks: ClassBlockTuple,
) -> NDArray[np.object_]:
    """Decode one label per sample and output from multioutput forest probability blocks."""
    n_samples = int(np.asarray(probability_blocks[0], dtype=np.float64).shape[0])
    predictions = np.empty((n_samples, len(probability_blocks)), dtype=object)
    for column, (probabilities, classes) in enumerate(zip(probability_blocks, classes_blocks)):
        matrix = np.asarray(probabilities, dtype=np.float64)
        class_values = np.asarray(classes, dtype=object)
        predictions[:, column] = class_values.take(np.argmax(matrix, axis=1), axis=0)
    return np.asarray(predictions, dtype=object)
