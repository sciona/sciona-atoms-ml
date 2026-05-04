"""Ghost witnesses for permutation feature multimetric aggregation atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_permutation_importance_feature_metric_names(
    score_dicts: tuple[dict[str, float], ...],
) -> tuple[str, ...]:
    """Describe the metric-name iteration order for one feature's score dicts."""
    if len(score_dicts) < 1:
        raise ValueError("score_dicts must be nonempty")
    if not score_dicts[0]:
        raise ValueError("first score dict must be nonempty")
    return tuple(score_dicts[0].keys())


def witness_permutation_importance_feature_metric_score_dict(
    score_dicts: tuple[dict[str, float], ...],
) -> dict[str, AbstractArray]:
    """Describe sklearn's dict-of-score-vectors aggregation for one feature."""
    if len(score_dicts) < 1:
        raise ValueError("score_dicts must be nonempty")
    if not score_dicts[0]:
        raise ValueError("first score dict must be nonempty")
    n_repeats = len(score_dicts)
    return {
        name: AbstractArray(shape=(n_repeats,), dtype="float64")
        for name in score_dicts[0]
    }
