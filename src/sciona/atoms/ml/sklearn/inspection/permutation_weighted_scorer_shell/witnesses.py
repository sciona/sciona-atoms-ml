"""Ghost witnesses for permutation weighted-scorer shell atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_permutation_importance_use_sample_weight(
    sample_weight: AbstractArray,
) -> AbstractArray:
    """Describe the sample-weight branch predicate in sklearn's _weights_scorer."""
    del sample_weight
    return AbstractArray(shape=(), dtype="bool")


def witness_permutation_importance_scorer_kwargs(
    sample_weight: AbstractArray,
) -> AbstractArray:
    """Describe the scorer kwargs mapping built by sklearn's _weights_scorer."""
    del sample_weight
    return AbstractArray(shape=(), dtype="object")
