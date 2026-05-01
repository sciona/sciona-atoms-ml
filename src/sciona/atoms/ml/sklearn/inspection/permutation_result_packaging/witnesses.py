"""Ghost witnesses for sklearn permutation-importance result packaging."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_permutation_importance_random_seed(random_state: int | None = None) -> int:
    """Describe sklearn's derived per-call permutation random seed."""
    del random_state
    return 0


def witness_permutation_importance_metric_score_matrix(
    score_dicts_by_feature: tuple[dict[str, AbstractArray], ...],
    metric_name: str,
) -> AbstractArray:
    """Describe per-feature stacked score vectors for one metric."""
    if len(score_dicts_by_feature) < 1:
        raise ValueError("score_dicts_by_feature must be nonempty")
    if not isinstance(metric_name, str) or len(metric_name) < 1:
        raise ValueError("metric_name must be nonempty")
    first = score_dicts_by_feature[0]
    if metric_name not in first:
        raise ValueError("metric_name missing from score dictionaries")
    n_repeats = int(first[metric_name].shape[0])
    return AbstractArray(shape=(len(score_dicts_by_feature), n_repeats), dtype="float64")


def witness_permutation_importance_summary_bunch(
    baseline_score: float,
    permuted_scores: AbstractArray,
) -> dict:
    """Describe sklearn's importances Bunch for one metric."""
    del baseline_score
    if len(permuted_scores.shape) != 2:
        raise ValueError("permuted_scores must be 2D")
    n_features = int(permuted_scores.shape[0])
    n_repeats = int(permuted_scores.shape[1])
    if n_features < 1 or n_repeats < 1:
        raise ValueError("permuted_scores must be nonempty")
    return {
        "importances_mean": AbstractArray(shape=(n_features,), dtype="float64"),
        "importances_std": AbstractArray(shape=(n_features,), dtype="float64"),
        "importances": AbstractArray(shape=(n_features, n_repeats), dtype="float64"),
    }


def witness_permutation_importance_multi_metric_bunches(
    baseline_scores: dict[str, float],
    score_dicts_by_feature: tuple[dict[str, AbstractArray], ...],
) -> dict[str, dict]:
    """Describe sklearn's dict-of-Bunch outputs for multimetric permutation importance."""
    if len(score_dicts_by_feature) < 1:
        raise ValueError("score_dicts_by_feature must be nonempty")
    if not baseline_scores:
        raise ValueError("baseline_scores must be nonempty")
    return {name: {} for name in baseline_scores}
