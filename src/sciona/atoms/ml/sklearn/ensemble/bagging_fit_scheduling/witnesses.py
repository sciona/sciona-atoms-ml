"""Ghost witnesses for sklearn bagging fit scheduling helpers."""

from __future__ import annotations


def witness_bagging_partition_estimators(
    n_estimators: int,
    n_jobs: int | None,
) -> tuple[int, object, object]:
    """Describe sklearn's balanced partition of estimators across jobs."""
    if n_estimators < 1:
        raise ValueError("n_estimators must be positive")
    return 0, (), ()


def witness_bagging_fit_seeds(
    n_more_estimators: int,
    previous_estimators: int,
    random_state: int | object | None = None,
) -> object:
    """Describe sklearn's warm-start-adjusted bagging seed generation."""
    if n_more_estimators < 1 or previous_estimators < 0:
        raise ValueError("estimator counts must be nonnegative and n_more_estimators positive")
    return random_state
