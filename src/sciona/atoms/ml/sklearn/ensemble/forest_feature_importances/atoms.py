"""Functions for forest feature importances."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_forest_average_feature_importances,
    witness_forest_importance_contributor_mask,
    witness_forest_normalized_feature_importances,
    witness_forest_zero_feature_importances,
)

FeatureImportanceBlock = NDArray[np.float64]
FeatureImportanceBlockTuple = tuple[FeatureImportanceBlock, ...]


def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _node_counts_valid(node_counts: object) -> bool:
    values = np.asarray(node_counts)
    return bool(
        values.ndim == 1
        and values.shape[0] >= 1
        and np.issubdtype(values.dtype, np.integer)
        and np.all(values >= 1)
    )


def _mask_valid(result: object, node_counts: object) -> bool:
    values = np.asarray(result)
    counts = np.asarray(node_counts)
    return bool(values.shape == counts.shape and values.dtype == np.bool_)


def _feature_importance_vector_valid(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 1
        and array.shape[0] >= 1
        and np.all(np.isfinite(array))
        and np.all(array >= 0.0)
    )


def _feature_importance_blocks_valid(blocks: object) -> bool:
    if not isinstance(blocks, tuple) or len(blocks) < 1:
        return False
    widths = []
    for block in blocks:
        if not _feature_importance_vector_valid(block):
            return False
        widths.append(int(np.asarray(block, dtype=np.float64).shape[0]))
    return len(set(widths)) == 1


def _zero_vector_valid(result: object, n_features: int) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(
        values.shape == (n_features,)
        and np.all(np.isfinite(values))
        and np.all(values == 0.0)
    )


def _average_vector_valid(result: object, feature_importance_blocks: FeatureImportanceBlockTuple) -> bool:
    values = np.asarray(result, dtype=np.float64)
    width = int(np.asarray(feature_importance_blocks[0], dtype=np.float64).shape[0])
    return bool(values.shape == (width,) and np.all(np.isfinite(values)) and np.all(values >= 0.0))


def _normalized_vector_input_valid(average_feature_importances: object) -> bool:
    if not _feature_importance_vector_valid(average_feature_importances):
        return False
    return bool(float(np.sum(np.asarray(average_feature_importances, dtype=np.float64))) > 0.0)


def _normalized_vector_valid(result: object, average_feature_importances: object) -> bool:
    values = np.asarray(result, dtype=np.float64)
    average = np.asarray(average_feature_importances, dtype=np.float64)
    return bool(
        values.shape == average.shape
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and np.isclose(np.sum(values), 1.0)
    )


@register_atom(witness_forest_importance_contributor_mask)
@icontract.require(
    lambda node_counts: _node_counts_valid(node_counts),
    "node_counts must be a nonempty vector of positive integers",
)
@icontract.ensure(
    lambda result, node_counts: _mask_valid(result, node_counts),
    "contributor mask must be a boolean vector aligned with node_counts",
)
def forest_importance_contributor_mask(
    node_counts: NDArray[np.int64],
) -> NDArray[np.bool_]:
    """Mark trees whose node counts make them contribute to sklearn forest feature importances."""
    counts = np.asarray(node_counts, dtype=np.int64)
    return np.asarray(counts > 1, dtype=np.bool_)


@register_atom(witness_forest_zero_feature_importances)
@icontract.require(
    lambda n_features: _positive_int(n_features),
    "n_features must be a positive integer",
)
@icontract.ensure(
    lambda result, n_features: _zero_vector_valid(result, n_features),
    "zero feature importances must be an all-zero vector of length n_features",
)
def forest_zero_feature_importances(
    n_features: int,
) -> NDArray[np.float64]:
    """Return sklearn's all-zero fallback when no tree contributes feature importances."""
    return np.zeros(n_features, dtype=np.float64)


@register_atom(witness_forest_average_feature_importances)
@icontract.require(
    lambda feature_importance_blocks: _feature_importance_blocks_valid(feature_importance_blocks),
    "feature_importance_blocks must be a nonempty tuple of aligned nonnegative finite vectors",
)
@icontract.ensure(
    lambda result, feature_importance_blocks: _average_vector_valid(result, feature_importance_blocks),
    "average feature importances must be a finite nonnegative vector aligned with the input blocks",
)
def forest_average_feature_importances(
    feature_importance_blocks: FeatureImportanceBlockTuple,
) -> NDArray[np.float64]:
    """Average contributing forest feature-importance vectors across trees."""
    stacked = np.stack([np.asarray(block, dtype=np.float64) for block in feature_importance_blocks], axis=0)
    return np.asarray(np.mean(stacked, axis=0, dtype=np.float64), dtype=np.float64)


@register_atom(witness_forest_normalized_feature_importances)
@icontract.require(
    lambda average_feature_importances: _normalized_vector_input_valid(average_feature_importances),
    "average_feature_importances must be a nonnegative finite vector with positive total mass",
)
@icontract.ensure(
    lambda result, average_feature_importances: _normalized_vector_valid(result, average_feature_importances),
    "normalized feature importances must be a finite nonnegative vector summing to 1",
)
def forest_normalized_feature_importances(
    average_feature_importances: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Normalize averaged forest feature importances to sklearn's public scale."""
    values = np.asarray(average_feature_importances, dtype=np.float64)
    return np.asarray(values / np.sum(values), dtype=np.float64)
