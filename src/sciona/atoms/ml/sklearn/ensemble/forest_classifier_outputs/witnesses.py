"""Ghost witnesses for sklearn forest classifier output helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_forest_classifier_log_probability_matrix(
    probabilities: AbstractArray,
) -> AbstractArray:
    """Describe log probabilities from a single-output forest classifier matrix."""
    if len(probabilities.shape) != 2:
        raise ValueError("probabilities must be a matrix")
    if int(probabilities.shape[0]) < 1 or int(probabilities.shape[1]) < 1:
        raise ValueError("probabilities must be nonempty")
    return AbstractArray(shape=probabilities.shape, dtype="float64")


def witness_forest_classifier_log_probability_blocks(
    probability_blocks: tuple[AbstractArray, ...],
) -> tuple[AbstractArray, ...]:
    """Describe log probabilities from a multioutput forest classifier sequence."""
    if len(probability_blocks) < 1:
        raise ValueError("probability_blocks must be nonempty")
    outputs: list[AbstractArray] = []
    for block in probability_blocks:
        if len(block.shape) != 2:
            raise ValueError("each probability block must be a matrix")
        outputs.append(AbstractArray(shape=block.shape, dtype="float64"))
    return tuple(outputs)


def witness_forest_classifier_multioutput_labels(
    probability_blocks: tuple[AbstractArray, ...],
    classes_blocks: tuple[AbstractArray, ...],
) -> AbstractArray:
    """Describe one predicted label per sample and output for forest multioutput classification."""
    if len(probability_blocks) < 1 or len(probability_blocks) != len(classes_blocks):
        raise ValueError("probability_blocks and classes_blocks must be nonempty and aligned")
    n_samples = int(probability_blocks[0].shape[0])
    return AbstractArray(shape=(n_samples, len(probability_blocks)), dtype="object")
