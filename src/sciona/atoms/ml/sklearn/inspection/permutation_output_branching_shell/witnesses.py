"""Ghost witnesses for permutation output-branching shell atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_permutation_importance_use_multimetric_results(
    baseline_score: AbstractArray,
) -> AbstractArray:
    """Describe the multimetric return-branch predicate in permutation_importance."""
    del baseline_score
    return AbstractArray(shape=(), dtype="bool")


def witness_permutation_importance_metric_names(
    baseline_scores: dict[str, float],
) -> tuple[str, ...]:
    """Describe the metric-name iteration order taken from baseline scores."""
    if not baseline_scores:
        raise ValueError("baseline_scores must be nonempty")
    return tuple(baseline_scores.keys())


def witness_permutation_importance_single_metric_score_matrix(
    scores: AbstractArray,
) -> AbstractArray:
    """Describe the single-metric score matrix coercion in permutation_importance."""
    if len(scores.shape) != 2:
        raise ValueError("scores must be two-dimensional")
    return AbstractArray(shape=scores.shape, dtype="float64")
