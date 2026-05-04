"""Ghost witnesses for permutation feature-score aggregation shell atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_permutation_importance_feature_scores_are_multimetric(
    first_score: AbstractArray,
) -> AbstractArray:
    """Describe the multimetric predicate for one feature's repeated scores."""
    del first_score
    return AbstractArray(shape=(), dtype="bool")


def witness_permutation_importance_single_feature_score_vector(
    scores: AbstractArray,
) -> AbstractArray:
    """Describe the one-feature repeated score vector coercion."""
    if len(scores.shape) != 1:
        raise ValueError("scores must be one-dimensional")
    return AbstractArray(shape=scores.shape, dtype="float64")
