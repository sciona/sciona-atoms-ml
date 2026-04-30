"""Witnesses for sklearn multiclass one-vs-rest probability bookkeeping helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_one_vs_rest_positive_probability_stack(
    probability_blocks: tuple[AbstractArray, ...],
) -> AbstractArray:
    """Describe positive-class probability extraction from per-estimator blocks."""
    if not isinstance(probability_blocks, tuple) or len(probability_blocks) < 1:
        raise ValueError("probability_blocks must be a nonempty tuple")
    n_samples = None
    for block in probability_blocks:
        if len(block.shape) != 2:
            raise ValueError("each probability block must be two-dimensional")
        rows = int(block.shape[0])
        cols = int(block.shape[1])
        if rows < 1 or cols < 2:
            raise ValueError("each probability block must be nonempty and have at least two columns")
        if n_samples is None:
            n_samples = rows
        elif rows != n_samples:
            raise ValueError("all probability blocks must share the same sample count")
    return AbstractArray(shape=(len(probability_blocks), int(n_samples)), dtype="float64")
