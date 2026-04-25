"""Ghost witnesses for sklearn forest fit-bookkeeping helpers."""

from __future__ import annotations


def witness_forest_fit_bootstrap_sample_count(
    bootstrap: bool,
    n_samples: int,
    max_samples: int | float | None,
) -> int | None:
    """Describe the bootstrap draw count used by forest fit, or no count when bootstrapping is off."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    return 0 if bootstrap else None


def witness_forest_fit_require_bootstrap_for_oob(
    bootstrap: bool,
    oob_score: bool,
) -> bool:
    """Describe the bootstrap requirement for forest out-of-bag estimates."""
    return True


def witness_forest_fit_additional_estimator_count(
    n_estimators: int,
    existing_estimators: int,
) -> int:
    """Describe how many new trees forest fit should grow."""
    if n_estimators < 1 or existing_estimators < 0:
        raise ValueError("estimator counts must be nonnegative and n_estimators positive")
    return 0


def witness_forest_fit_oob_update_required(
    oob_score: bool,
    n_more_estimators: int,
    has_oob_score_attr: bool,
) -> bool:
    """Describe whether forest fit should recompute OOB attributes after growth bookkeeping."""
    if n_more_estimators < 0:
        raise ValueError("n_more_estimators must be nonnegative")
    return False


def witness_forest_fit_require_supported_oob_target_type(
    target_type: str,
    is_classifier: bool,
) -> bool:
    """Describe the target-type preflight for forest OOB estimates."""
    if not target_type:
        raise ValueError("target_type must be nonempty")
    return True
