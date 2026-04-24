"""Ghost witnesses for sklearn bagging fit-shell bookkeeping helpers."""

from __future__ import annotations


def witness_bagging_resolve_max_samples(
    max_samples_override: int | float | None,
    configured_max_samples: int | float,
    n_samples: int,
) -> int:
    """Describe the validated integer sample count used by bagging fit."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    return 0


def witness_bagging_resolve_max_features(
    configured_max_features: int | float,
    n_features: int,
) -> int:
    """Describe the validated integer feature count used by bagging fit."""
    if n_features < 1:
        raise ValueError("n_features must be positive")
    return 0


def witness_bagging_fit_require_bootstrap_for_oob(
    bootstrap: bool,
    oob_score: bool,
) -> bool:
    """Describe the bootstrap requirement for out-of-bag estimation."""
    return True


def witness_bagging_fit_require_no_warm_start_with_oob(
    warm_start: bool,
    oob_score: bool,
) -> bool:
    """Describe the warm-start restriction for out-of-bag estimation."""
    return True


def witness_bagging_additional_estimator_count(
    n_estimators: int,
    existing_estimators: int,
) -> int:
    """Describe how many new estimators bagging fit should build."""
    if n_estimators < 1 or existing_estimators < 0:
        raise ValueError("estimator counts must be nonnegative and n_estimators positive")
    return 0
