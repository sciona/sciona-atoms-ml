"""Ghost witnesses for RFE fit-bookkeeping atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_rfe_resolve_n_features_to_select(
    n_features: int,
    *,
    n_features_to_select: int | float | None = None,
) -> int:
    """Describe one resolved RFE target feature count."""
    del n_features_to_select
    if n_features < 1:
        raise ValueError("n_features must be positive")
    return 1


def witness_rfe_warn_too_many_features_to_select(
    n_features: int,
    *,
    resolved_n_features_to_select: int,
) -> bool:
    """Describe the oversize-feature-count warning predicate."""
    if n_features < 1 or resolved_n_features_to_select < 1:
        raise ValueError("feature counts must be positive")
    return False


def witness_rfe_resolve_step(
    n_features: int,
    *,
    step: int | float = 1,
) -> int:
    """Describe one resolved RFE elimination step."""
    del step
    if n_features < 1:
        raise ValueError("n_features must be positive")
    return 1


def witness_rfe_active_feature_indices(
    support_mask: AbstractArray,
) -> AbstractArray:
    """Describe the active feature-index vector from an RFE support mask."""
    if len(support_mask.shape) != 1:
        raise ValueError("support_mask must be 1D")
    n_features = int(support_mask.shape[0])
    if n_features < 1:
        raise ValueError("support_mask must be nonempty")
    return AbstractArray(shape=(n_features,), dtype="int64", min_val=0.0)


def witness_rfe_step_history_append(
    step_n_features: AbstractArray,
    step_scores: AbstractArray,
    *,
    n_features: int,
    score: float,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe an RFE step-history append for feature counts and scores."""
    del score
    if len(step_n_features.shape) != 1 or len(step_scores.shape) != 1:
        raise ValueError("history arrays must be 1D")
    if step_n_features.shape != step_scores.shape:
        raise ValueError("history arrays must have matching lengths")
    if n_features < 1:
        raise ValueError("n_features must be positive")
    next_len = int(step_n_features.shape[0]) + 1
    return (
        AbstractArray(shape=(next_len,), dtype="int64", min_val=1.0),
        AbstractArray(shape=(next_len,), dtype="float64"),
    )
