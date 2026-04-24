"""Ghost witnesses for deterministic RFECV aggregation atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _step_count(step_scores: AbstractArray) -> tuple[int, int]:
    if len(step_scores.shape) != 2:
        raise ValueError("step_scores must be 2D")
    n_folds = int(step_scores.shape[0])
    n_steps = int(step_scores.shape[1])
    if n_folds < 1 or n_steps < 1:
        raise ValueError("step_scores must be nonempty")
    return n_folds, n_steps


def _validate_step_path(step_n_features: AbstractArray, n_folds: int, n_steps: int) -> None:
    if len(step_n_features.shape) == 1:
        if int(step_n_features.shape[0]) != n_steps:
            raise ValueError("1D step_n_features must match the step axis")
        return
    if len(step_n_features.shape) == 2:
        if tuple(step_n_features.shape) != (n_folds, n_steps):
            raise ValueError("2D step_n_features must match the fold and step axes")
        return
    raise ValueError("step_n_features must be 1D or 2D")


def witness_rfecv_best_feature_count(
    step_scores: AbstractArray,
    step_n_features: AbstractArray,
) -> AbstractArray:
    """Describe the selected RFECV feature count after reversed tie-breaking."""
    n_folds, n_steps = _step_count(step_scores)
    _validate_step_path(step_n_features, n_folds, n_steps)
    return AbstractArray(shape=(), dtype="int64")


def witness_rfecv_cv_results(
    step_scores: AbstractArray,
    step_n_features: AbstractArray,
) -> dict[str, AbstractArray]:
    """Describe RFECV cv_results_ arrays after reversing the elimination path."""
    n_folds, n_steps = _step_count(step_scores)
    _validate_step_path(step_n_features, n_folds, n_steps)
    return {
        "mean_test_score": AbstractArray(shape=(n_steps,), dtype="float64"),
        "std_test_score": AbstractArray(shape=(n_steps,), dtype="float64"),
        **{f"split{i}_test_score": AbstractArray(shape=(n_steps,), dtype="float64") for i in range(n_folds)},
        "n_features": AbstractArray(shape=(n_steps,), dtype="int64"),
    }
