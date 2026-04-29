"""Ghost witnesses for MLP stochastic-batching helper atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_mlp_stochastic_stratify_targets(
    y: AbstractArray,
    *,
    is_classifier: bool,
    n_outputs: int,
) -> AbstractArray | None:
    """Describe sklearn's stratify argument for the early-stopping split."""
    if len(y.shape) not in {1, 2} or int(y.shape[0]) < 1:
        raise ValueError("y must be a nonempty 1D or 2D array")
    if n_outputs < 1:
        raise ValueError("n_outputs must be positive")
    if is_classifier and n_outputs == 1:
        return y
    return None


def witness_mlp_stochastic_sample_indices(n_samples: int) -> AbstractArray:
    """Describe the per-epoch zero-based sample index vector."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    return AbstractArray(shape=(n_samples,), dtype="int64")


def witness_mlp_stochastic_batches_per_epoch(n_samples: int, batch_size: int) -> int:
    """Describe the number of minibatches processed in one epoch."""
    if n_samples < 1 or batch_size < 1:
        raise ValueError("n_samples and batch_size must be positive")
    return 1


def witness_mlp_stochastic_batch_indices(
    sample_indices: AbstractArray,
    *,
    batch_start: int,
    batch_stop: int,
    shuffle: bool,
) -> AbstractArray:
    """Describe one minibatch's selected sample indices."""
    del shuffle
    if len(sample_indices.shape) != 1 or int(sample_indices.shape[0]) < 1:
        raise ValueError("sample_indices must be a nonempty 1D array")
    if batch_start < 0 or batch_stop <= batch_start or batch_stop > int(sample_indices.shape[0]):
        raise ValueError("batch bounds must select a nonempty in-range minibatch")
    return AbstractArray(shape=(batch_stop - batch_start,), dtype="int64")


def witness_mlp_stochastic_accumulated_loss(
    accumulated_loss: float,
    batch_loss: float,
    *,
    batch_start: int,
    batch_stop: int,
) -> float:
    """Describe minibatch-size-weighted loss accumulation."""
    del accumulated_loss, batch_loss
    if batch_start < 0 or batch_stop <= batch_start:
        raise ValueError("batch bounds must select a nonempty minibatch")
    return 0.0
