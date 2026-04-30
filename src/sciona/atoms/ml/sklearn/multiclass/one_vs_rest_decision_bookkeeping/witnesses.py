"""Ghost witnesses for one-vs-rest decision-function bookkeeping atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_one_vs_rest_decision_stack(
    decision_blocks: tuple[AbstractArray, ...],
) -> AbstractArray:
    """Describe the estimator-by-sample decision stack built before OvR output shaping."""
    if len(decision_blocks) < 1:
        raise ValueError("decision_blocks must be nonempty")
    first = decision_blocks[0]
    if len(first.shape) != 1:
        raise ValueError("decision blocks must be vectors")
    n_samples = int(first.shape[0])
    if n_samples < 1:
        raise ValueError("decision blocks must contain at least one sample")
    return AbstractArray(shape=(len(decision_blocks), n_samples), dtype="float64")
