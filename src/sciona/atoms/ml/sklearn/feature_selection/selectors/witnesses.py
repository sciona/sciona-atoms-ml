"""Ghost witnesses for sklearn estimator-callback selector helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray

ThresholdSpec = str | float | None
TransformSpec = str | None


def _check_vector(values: AbstractArray, name: str) -> int:
    if len(values.shape) != 1:
        raise ValueError(f"{name} must be 1D")
    size = int(values.shape[0])
    if size < 1:
        raise ValueError(f"{name} must be nonempty")
    return size


def witness_feature_importances_transform(
    importances: AbstractArray,
    *,
    transform_func: TransformSpec = None,
    norm_order: int = 1,
) -> AbstractArray:
    """Describe a one-score-per-feature importance vector after transformation."""
    del norm_order
    if transform_func not in {None, "norm", "square"}:
        raise ValueError("unsupported transform function")
    if len(importances.shape) == 1:
        return AbstractArray(shape=(int(importances.shape[0]),), dtype="float64")
    if len(importances.shape) == 2:
        return AbstractArray(shape=(int(importances.shape[1]),), dtype="float64")
    raise ValueError("importances must be 1D or 2D")


def witness_select_from_model_threshold(
    importances: AbstractArray,
    *,
    threshold: ThresholdSpec = None,
    l1_default: bool = False,
) -> AbstractArray:
    """Describe a scalar SelectFromModel threshold from score statistics."""
    del threshold, l1_default
    _check_vector(importances, "importances")
    return AbstractArray(shape=(), dtype="float64")


def witness_select_from_model_support_mask(
    scores: AbstractArray,
    *,
    threshold: float,
    max_features: int | None = None,
) -> AbstractArray:
    """Describe a support mask from scores, threshold, and optional feature cap."""
    del threshold, max_features
    n_features = _check_vector(scores, "scores")
    return AbstractArray(shape=(n_features,), dtype="bool")


def witness_rfe_elimination_step(
    support_mask: AbstractArray,
    ranking: AbstractArray,
    importances: AbstractArray,
    *,
    n_features_to_select: int,
    step: int,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe one RFE support/ranking update from supplied importances."""
    del n_features_to_select, step
    n_features = _check_vector(support_mask, "support_mask")
    if _check_vector(ranking, "ranking") != n_features:
        raise ValueError("ranking length must match support mask")
    _check_vector(importances, "importances")
    return (
        AbstractArray(shape=(n_features,), dtype="bool"),
        AbstractArray(shape=(n_features,), dtype="int64"),
    )


def witness_sequential_candidate_masks(
    current_mask: AbstractArray,
    *,
    direction: str = "forward",
) -> tuple[AbstractArray, AbstractArray]:
    """Describe candidate masks for one sequential feature-selection step."""
    if direction not in {"forward", "backward"}:
        raise ValueError("direction must be forward or backward")
    n_features = _check_vector(current_mask, "current_mask")
    return (
        AbstractArray(shape=(n_features,), dtype="int64"),
        AbstractArray(shape=(n_features, n_features), dtype="bool"),
    )


def witness_sequential_best_feature(
    candidate_indices: AbstractArray,
    scores: AbstractArray,
) -> AbstractArray:
    """Describe the selected candidate index with largest supplied score."""
    n_candidates = _check_vector(candidate_indices, "candidate_indices")
    if _check_vector(scores, "scores") != n_candidates:
        raise ValueError("scores length must match candidate count")
    return AbstractArray(shape=(), dtype="int64")
