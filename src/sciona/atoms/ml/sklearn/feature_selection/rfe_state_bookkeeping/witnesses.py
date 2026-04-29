"""Ghost witnesses for RFE state-bookkeeping atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_rfe_initial_support_mask(
    n_features: int,
) -> AbstractArray:
    """Describe the all-active support mask at the start of RFE fitting."""
    if n_features < 1:
        raise ValueError("n_features must be positive")
    return AbstractArray(shape=(n_features,), dtype="bool")


def witness_rfe_initial_ranking(
    n_features: int,
) -> AbstractArray:
    """Describe the all-ones ranking vector at the start of RFE fitting."""
    if n_features < 1:
        raise ValueError("n_features must be positive")
    return AbstractArray(shape=(n_features,), dtype="int64", min_val=1.0)


def witness_rfe_initial_step_history() -> tuple[AbstractArray, AbstractArray]:
    """Describe empty step history vectors for RFE."""
    return (
        AbstractArray(shape=(0,), dtype="int64"),
        AbstractArray(shape=(0,), dtype="float64"),
    )


def witness_rfe_elimination_threshold(
    active_feature_count: int,
    *,
    n_features_to_select: int,
    step: int,
) -> int:
    """Describe the bounded elimination count for one RFE iteration."""
    del n_features_to_select, step
    if active_feature_count < 2:
        raise ValueError("active_feature_count must allow at least one elimination")
    return 1


def witness_rfe_final_feature_count(
    support_mask: AbstractArray,
) -> int:
    """Describe the final selected-feature count from an RFE support mask."""
    if len(support_mask.shape) != 1 or int(support_mask.shape[0]) < 1:
        raise ValueError("support_mask must be a nonempty 1D vector")
    return 1
