"""Ghost witnesses for sklearn multioutput classifier output bookkeeping helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_multioutput_predict_require_base_predict_method(estimator_has_predict: bool) -> bool:
    """Describe the base-estimator predict-method guard for multioutput predict."""
    if not isinstance(estimator_has_predict, bool):
        raise ValueError("estimator_has_predict must be boolean")
    return estimator_has_predict


def witness_multioutput_classifier_score_require_2d_targets(y: AbstractArray) -> AbstractArray:
    """Describe multioutput classifier score targets after the 2D guard."""
    if len(y.shape) == 1:
        raise ValueError("y must be 2D for multioutput classifier scoring")
    if len(y.shape) != 2:
        raise ValueError("y must be 2D")
    n_samples = int(y.shape[0])
    n_outputs = int(y.shape[1])
    if n_samples < 1 or n_outputs < 1:
        raise ValueError("y must be nonempty")
    return AbstractArray(shape=(n_samples, n_outputs), dtype="float64")


def witness_multioutput_classifier_score_require_matching_output_count(
    y: AbstractArray,
    n_outputs: int,
) -> AbstractArray:
    """Describe multioutput classifier score targets after the output-count guard."""
    if len(y.shape) != 2:
        raise ValueError("y must be 2D")
    n_samples = int(y.shape[0])
    y_outputs = int(y.shape[1])
    if n_samples < 1 or y_outputs < 1:
        raise ValueError("y must be nonempty")
    if n_outputs != y_outputs:
        raise ValueError("n_outputs must match the number of target columns")
    return AbstractArray(shape=(n_samples, n_outputs), dtype="float64")


def witness_multioutput_classifier_probability_blocks(
    probability_blocks: tuple[AbstractArray, ...],
) -> tuple[AbstractArray, ...]:
    """Describe the per-output probability matrices returned by multioutput classification."""
    if len(probability_blocks) < 1:
        raise ValueError("probability_blocks must be nonempty")
    outputs: list[AbstractArray] = []
    for block in probability_blocks:
        if len(block.shape) != 2:
            raise ValueError("each probability block must be 2D")
        n_samples = int(block.shape[0])
        n_classes = int(block.shape[1])
        if n_samples < 1 or n_classes < 1:
            raise ValueError("each probability block must be nonempty")
        outputs.append(AbstractArray(shape=(n_samples, n_classes), dtype="float64"))
    return tuple(outputs)
