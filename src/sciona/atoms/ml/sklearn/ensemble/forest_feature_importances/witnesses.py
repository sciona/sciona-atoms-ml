"""Ghost witnesses for sklearn forest feature-importance helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_forest_importance_contributor_mask(
    node_counts: AbstractArray,
) -> AbstractArray:
    """Describe which forest trees contribute feature importances."""
    if len(node_counts.shape) != 1 or int(node_counts.shape[0]) < 1:
        raise ValueError("node_counts must be a nonempty vector")
    return AbstractArray(shape=node_counts.shape, dtype="bool")


def witness_forest_zero_feature_importances(
    n_features: int,
) -> AbstractArray:
    """Describe the all-zero importance vector when no tree contributes."""
    if n_features < 1:
        raise ValueError("n_features must be positive")
    return AbstractArray(shape=(n_features,), dtype="float64", min_val=0.0, max_val=0.0)


def witness_forest_average_feature_importances(
    feature_importance_blocks: tuple[AbstractArray, ...],
) -> AbstractArray:
    """Describe the mean feature-importance vector across contributing trees."""
    if len(feature_importance_blocks) < 1:
        raise ValueError("feature_importance_blocks must be nonempty")
    width = int(feature_importance_blocks[0].shape[0])
    if width < 1:
        raise ValueError("feature importance vectors must be nonempty")
    return AbstractArray(shape=(width,), dtype="float64", min_val=0.0)


def witness_forest_normalized_feature_importances(
    average_feature_importances: AbstractArray,
) -> AbstractArray:
    """Describe the normalized forest feature-importance vector."""
    if len(average_feature_importances.shape) != 1 or int(average_feature_importances.shape[0]) < 1:
        raise ValueError("average_feature_importances must be a nonempty vector")
    return AbstractArray(shape=average_feature_importances.shape, dtype="float64", min_val=0.0)
