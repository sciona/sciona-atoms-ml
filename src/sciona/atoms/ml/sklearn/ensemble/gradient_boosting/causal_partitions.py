"""Source recipe adapter: expose five training partitions without fitting."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom
from .causal_training import causal_training_partitions

ROLES = ('one_step', 'independence', 'direction', 'left', 'right')


def witness_prepare_causal_training_partitions(training_features, training_labels) -> tuple:
    # Direction rows depend on label values. Shape metadata alone cannot supply
    # their cardinality; never invent a fixed size or use all rows instead.
    if isinstance(training_features, AbstractArray) or isinstance(training_labels, AbstractArray):
        raise ValueError('Concrete labels/features required for data-dependent partition witness')
    parts = causal_training_partitions(training_features, training_labels)
    return tuple(None if value is None else AbstractArray(shape=value.shape,dtype=str(value.dtype))
                 for part in parts for value in part)


@register_atom(witness_prepare_causal_training_partitions)
def prepare_causal_training_partitions(
    training_features: NDArray[np.float64], training_labels: NDArray[np.int64],
) -> tuple:
    """Return explicit (features, labels, weights) ports for each source role.

    Flattened order is one-step, independence, direction, left, right. Uses
    the reviewed source partition rules, including adjacent reversed labels,
    exclusion of independent rows from direction training, unit direction
    weights and negative-class balancing for the three binary indicator tasks.
    One-step weights remain None. No fitting or prediction occurs here.
    This is a recipe adapter; the downstream weighted fitter is generic.
    All arrays are copied so one consumer cannot mutate another role's inputs.
    Abstract-only shape propagation is unsupported because direction row count
    depends on actual labels; concrete execution remains supported.
    """
    return tuple(None if value is None else value.copy()
                 for part in causal_training_partitions(training_features,training_labels)
                 for value in part)
