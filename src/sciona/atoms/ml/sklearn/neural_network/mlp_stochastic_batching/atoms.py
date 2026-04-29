"""MLP stochastic-batching helper atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_mlp_stochastic_accumulated_loss,
    witness_mlp_stochastic_batch_indices,
    witness_mlp_stochastic_batches_per_epoch,
    witness_mlp_stochastic_sample_indices,
    witness_mlp_stochastic_stratify_targets,
)

IndexVector = NDArray[np.int64]
TargetArray = NDArray[np.float64] | NDArray[np.bool_]


def _bool_valid(value: object) -> bool:
    return isinstance(value, bool)


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _target_array_valid(values: object) -> bool:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim in {1, 2}
        and array.shape[0] >= 1
        and (
            np.issubdtype(array.dtype, np.number)
            or np.issubdtype(array.dtype, np.bool_)
        )
        and np.all(np.isfinite(array.astype(np.float64, copy=False)))
    )


def _index_vector_valid(values: object) -> bool:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 1
        and array.shape[0] >= 1
        and np.issubdtype(array.dtype, np.integer)
        and np.array_equal(array, np.arange(array.shape[0], dtype=array.dtype))
    )


def _integer_vector_valid(values: object) -> bool:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 1
        and array.shape[0] >= 1
        and np.issubdtype(array.dtype, np.integer)
    )


def _batch_bounds_valid(sample_indices: object, batch_start: int, batch_stop: int) -> bool:
    try:
        indices = np.asarray(sample_indices)
    except (TypeError, ValueError):
        return False
    return bool(
        indices.ndim == 1
        and indices.shape[0] >= 1
        and _nonnegative_int(batch_start)
        and _positive_int(batch_stop)
        and int(batch_start) < int(batch_stop)
        and int(batch_stop) <= indices.shape[0]
    )


def _finite_scalar(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value))


def _stratify_result_valid(result: object, y: object, is_classifier: bool, n_outputs: int) -> bool:
    if is_classifier and n_outputs == 1:
        try:
            return np.array_equal(np.asarray(result), np.asarray(y))
        except (TypeError, ValueError):
            return False
    return result is None


def _batch_indices_valid(
    result: object,
    sample_indices: object,
    batch_start: int,
    batch_stop: int,
    shuffle: bool,
) -> bool:
    try:
        values = np.asarray(result)
        source = np.asarray(sample_indices)
    except (TypeError, ValueError):
        return False
    if not (values.ndim == 1 and np.issubdtype(values.dtype, np.integer)):
        return False
    expected = (
        source[int(batch_start) : int(batch_stop)]
        if shuffle
        else np.arange(int(batch_start), int(batch_stop), dtype=np.int64)
    )
    return bool(np.array_equal(values, expected))


@register_atom(witness_mlp_stochastic_stratify_targets)
@icontract.require(lambda y: _target_array_valid(y), "y must be a finite nonempty numeric or boolean target array")
@icontract.require(lambda is_classifier: _bool_valid(is_classifier), "is_classifier must be boolean")
@icontract.require(lambda n_outputs: _positive_int(n_outputs), "n_outputs must be a positive integer")
@icontract.ensure(
    lambda result, y, is_classifier, n_outputs: _stratify_result_valid(result, y, is_classifier, n_outputs),
    "stratify targets must equal y for single-output classifiers and None otherwise",
)
def mlp_stochastic_stratify_targets(
    y: TargetArray,
    *,
    is_classifier: bool,
    n_outputs: int,
) -> TargetArray | None:
    """Resolve sklearn's early-stopping stratify argument inside `_fit_stochastic`."""
    return np.asarray(y).copy() if is_classifier and int(n_outputs) == 1 else None


@register_atom(witness_mlp_stochastic_sample_indices)
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be a positive integer")
@icontract.ensure(lambda result: _index_vector_valid(result), "sample indices must be a zero-based contiguous integer vector")
def mlp_stochastic_sample_indices(
    n_samples: int,
) -> IndexVector:
    """Create sklearn's per-epoch sample-index vector before optional shuffling."""
    return np.arange(int(n_samples), dtype=np.int64)


@register_atom(witness_mlp_stochastic_batches_per_epoch)
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be a positive integer")
@icontract.require(lambda batch_size: _positive_int(batch_size), "batch_size must be a positive integer")
@icontract.ensure(lambda result: _positive_int(result), "batches per epoch must be a positive integer")
def mlp_stochastic_batches_per_epoch(
    n_samples: int,
    batch_size: int,
) -> int:
    """Compute how many minibatches sklearn processes in one stochastic epoch."""
    return int(np.ceil(int(n_samples) / int(batch_size)))


@register_atom(witness_mlp_stochastic_batch_indices)
@icontract.require(lambda sample_indices: _integer_vector_valid(sample_indices), "sample_indices must be a nonempty integer vector")
@icontract.require(
    lambda sample_indices, batch_start, batch_stop: _batch_bounds_valid(sample_indices, batch_start, batch_stop),
    "batch_start and batch_stop must select a nonempty in-range minibatch",
)
@icontract.require(lambda shuffle: _bool_valid(shuffle), "shuffle must be boolean")
@icontract.ensure(
    lambda result, sample_indices, batch_start, batch_stop, shuffle: _batch_indices_valid(
        result, sample_indices, batch_start, batch_stop, shuffle
    ),
    "batch indices must match sklearn's shuffled or contiguous minibatch selection",
)
def mlp_stochastic_batch_indices(
    sample_indices: IndexVector,
    *,
    batch_start: int,
    batch_stop: int,
    shuffle: bool,
) -> IndexVector:
    """Select one minibatch index vector from sklearn's epoch sample-order state."""
    if shuffle:
        return np.asarray(sample_indices[int(batch_start) : int(batch_stop)], dtype=np.int64)
    return np.arange(int(batch_start), int(batch_stop), dtype=np.int64)


@register_atom(witness_mlp_stochastic_accumulated_loss)
@icontract.require(lambda accumulated_loss: _finite_scalar(accumulated_loss), "accumulated_loss must be finite")
@icontract.require(lambda batch_loss: _finite_scalar(batch_loss), "batch_loss must be finite")
@icontract.require(lambda batch_start: _nonnegative_int(batch_start), "batch_start must be a nonnegative integer")
@icontract.require(lambda batch_stop: _positive_int(batch_stop), "batch_stop must be a positive integer")
@icontract.require(lambda batch_start, batch_stop: int(batch_start) < int(batch_stop), "batch_start must be less than batch_stop")
@icontract.ensure(lambda result: _finite_scalar(result), "updated accumulated loss must be finite")
def mlp_stochastic_accumulated_loss(
    accumulated_loss: float,
    batch_loss: float,
    *,
    batch_start: int,
    batch_stop: int,
) -> float:
    """Accumulate sklearn's minibatch loss weighted by the minibatch sample count."""
    batch_count = int(batch_stop) - int(batch_start)
    return float(accumulated_loss) + float(batch_loss) * float(batch_count)
