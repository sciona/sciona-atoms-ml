"""Ghost witnesses for bagging aggregation helper atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_probability_blocks(
    probability_blocks: tuple[AbstractArray, ...],
    class_index_blocks: tuple[AbstractArray, ...],
    n_classes: int,
) -> int:
    if n_classes < 1:
        raise ValueError("n_classes must be positive")
    if len(probability_blocks) < 1 or len(probability_blocks) != len(class_index_blocks):
        raise ValueError("probability_blocks and class_index_blocks must be nonempty tuples with matching length")

    n_samples: int | None = None
    for block, class_indices in zip(probability_blocks, class_index_blocks):
        if len(block.shape) != 2:
            raise ValueError("each probability block must be 2D")
        if len(class_indices.shape) != 1:
            raise ValueError("each class-index block must be 1D")
        rows, cols = int(block.shape[0]), int(block.shape[1])
        if rows < 1 or cols < 1:
            raise ValueError("probability blocks must be nonempty")
        if int(class_indices.shape[0]) != cols:
            raise ValueError("each probability block must align with its class-index block")
        if n_samples is None:
            n_samples = rows
        elif rows != n_samples:
            raise ValueError("all probability blocks must share the same sample count")
    return int(n_samples)


def _check_same_shape_blocks(
    blocks: tuple[AbstractArray, ...],
    *,
    name: str,
) -> tuple[int, ...]:
    if len(blocks) < 1:
        raise ValueError(f"{name} must be nonempty")
    first = blocks[0]
    if len(first.shape) not in {1, 2}:
        raise ValueError(f"{name} blocks must be 1D or 2D")
    shape = tuple(int(dim) for dim in first.shape)
    if any(dim < 1 for dim in shape):
        raise ValueError(f"{name} blocks must be nonempty")
    for block in blocks[1:]:
        if tuple(int(dim) for dim in block.shape) != shape:
            raise ValueError(f"all {name} blocks must share the same shape")
    return shape


def witness_bagging_classifier_average_probabilities(
    probability_blocks: tuple[AbstractArray, ...],
    class_index_blocks: tuple[AbstractArray, ...],
    *,
    n_classes: int,
) -> AbstractArray:
    """Describe bagging probability averaging with class alignment."""
    n_samples = _check_probability_blocks(probability_blocks, class_index_blocks, n_classes)
    return AbstractArray(shape=(n_samples, n_classes), dtype="float64", min_val=0.0, max_val=1.0)


def witness_bagging_classifier_average_log_probabilities(
    log_probability_blocks: tuple[AbstractArray, ...],
    class_index_blocks: tuple[AbstractArray, ...],
    *,
    n_classes: int,
) -> AbstractArray:
    """Describe bagging log-probability averaging with class alignment."""
    n_samples = _check_probability_blocks(log_probability_blocks, class_index_blocks, n_classes)
    return AbstractArray(shape=(n_samples, n_classes), dtype="float64")


def witness_bagging_classifier_average_decision_function(
    decision_blocks: tuple[AbstractArray, ...],
) -> AbstractArray:
    """Describe averaging bagging decision-function outputs."""
    shape = _check_same_shape_blocks(decision_blocks, name="decision")
    return AbstractArray(shape=shape, dtype="float64")


def witness_bagging_regressor_average_predictions(
    prediction_blocks: tuple[AbstractArray, ...],
) -> AbstractArray:
    """Describe averaging bagging regressor predictions."""
    shape = _check_same_shape_blocks(prediction_blocks, name="prediction")
    return AbstractArray(shape=shape, dtype="float64")
