"""Birch fit and global-bookkeeping atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_birch_compute_labels_required,
    witness_birch_copy_warning_required,
    witness_birch_first_call,
    witness_birch_identity_subcluster_labels,
    witness_birch_leaf_centers,
    witness_birch_n_features_out,
    witness_birch_not_enough_centroids,
)

CentroidBlock = NDArray[np.float64]
CentroidBlockTuple = tuple[CentroidBlock, ...]


def _bool_value(value: object) -> bool:
    return isinstance(value, bool)


def _copy_value_valid(value: object) -> bool:
    return isinstance(value, bool) or value == "deprecated"


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _finite_centroid_block(block: object) -> bool:
    try:
        values = np.asarray(block, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 2 and values.shape[0] >= 1 and values.shape[1] >= 1 and np.all(np.isfinite(values)))


def _centroid_blocks_valid(blocks: object) -> bool:
    if not isinstance(blocks, tuple) or len(blocks) < 1:
        return False
    values = [np.asarray(block, dtype=np.float64) for block in blocks]
    if not all(_finite_centroid_block(block) for block in values):
        return False
    n_features = values[0].shape[1]
    return all(block.shape[1] == n_features for block in values)


def _concatenated_centers_valid(result: object, blocks: CentroidBlockTuple) -> bool:
    values = np.asarray(result, dtype=np.float64)
    total_rows = sum(np.asarray(block, dtype=np.float64).shape[0] for block in blocks)
    n_features = np.asarray(blocks[0], dtype=np.float64).shape[1]
    return bool(
        values.shape == (total_rows, n_features)
        and np.all(np.isfinite(values))
    )


def _identity_labels_valid(result: object, n_centroids: int) -> bool:
    values = np.asarray(result)
    return bool(
        values.shape == (n_centroids,)
        and np.issubdtype(values.dtype, np.integer)
        and np.array_equal(values, np.arange(n_centroids))
    )


def _finite_center_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


@register_atom(witness_birch_first_call)
@icontract.require(lambda partial: _bool_value(partial), "partial must be boolean")
@icontract.require(lambda has_root: _bool_value(has_root), "has_root must be boolean")
@icontract.ensure(lambda result, partial, has_root: result == (not (partial and has_root)), "first-call flag must match Birch._fit")
def birch_first_call(partial: bool, has_root: bool) -> bool:
    """Return whether Birch._fit should initialize a new CF tree."""
    return not (partial and has_root)


@register_atom(witness_birch_copy_warning_required)
@icontract.require(lambda copy: _copy_value_valid(copy), "copy must be boolean or 'deprecated'")
@icontract.require(lambda first_call: _bool_value(first_call), "first_call must be boolean")
@icontract.ensure(lambda result, copy, first_call: result == (copy != "deprecated" and first_call), "warning predicate must match Birch._fit")
def birch_copy_warning_required(copy: str | bool, first_call: bool) -> bool:
    """Return whether Birch should emit its copy-deprecation warning."""
    return bool(copy != "deprecated" and first_call)


@register_atom(witness_birch_compute_labels_required)
@icontract.require(lambda has_input_data: _bool_value(has_input_data), "has_input_data must be boolean")
@icontract.require(lambda compute_labels: _bool_value(compute_labels), "compute_labels must be boolean")
@icontract.ensure(lambda result, has_input_data, compute_labels: result == (has_input_data and compute_labels), "label refresh predicate must match Birch._global_clustering")
def birch_compute_labels_required(has_input_data: bool, compute_labels: bool) -> bool:
    """Return whether Birch._global_clustering should recompute labels_."""
    return bool(has_input_data and compute_labels)


@register_atom(witness_birch_not_enough_centroids)
@icontract.require(lambda n_centroids: _positive_int(n_centroids), "n_centroids must be positive")
@icontract.require(lambda n_clusters: _positive_int(n_clusters), "n_clusters must be positive")
@icontract.ensure(lambda result, n_centroids, n_clusters: result == (n_centroids < n_clusters), "predicate must match Birch integer-cluster short circuit")
def birch_not_enough_centroids(n_centroids: int, n_clusters: int) -> bool:
    """Return whether Birch should skip integer global clustering because too few subclusters exist."""
    return bool(n_centroids < n_clusters)


@register_atom(witness_birch_identity_subcluster_labels)
@icontract.require(lambda n_centroids: _positive_int(n_centroids), "n_centroids must be positive")
@icontract.ensure(lambda result, n_centroids: _identity_labels_valid(result, n_centroids), "identity labels must enumerate subclusters")
def birch_identity_subcluster_labels(n_centroids: int) -> NDArray[np.int64]:
    """Return Birch's identity subcluster labels for no-global-clustering branches."""
    return np.arange(n_centroids, dtype=np.int64)


@register_atom(witness_birch_leaf_centers)
@icontract.require(lambda leaf_centroid_blocks: _centroid_blocks_valid(leaf_centroid_blocks), "leaf_centroid_blocks must be a nonempty tuple of finite 2D centroid blocks with a shared feature count")
@icontract.ensure(lambda result, leaf_centroid_blocks: _concatenated_centers_valid(result, leaf_centroid_blocks), "concatenated centers must preserve total rows and feature count")
def birch_leaf_centers(leaf_centroid_blocks: CentroidBlockTuple) -> NDArray[np.float64]:
    """Concatenate Birch leaf-centroid blocks into the subcluster-centers matrix."""
    return np.asarray(
        np.concatenate(tuple(np.asarray(block, dtype=np.float64) for block in leaf_centroid_blocks)),
        dtype=np.float64,
    )


@register_atom(witness_birch_n_features_out)
@icontract.require(lambda subcluster_centers: _finite_center_matrix(subcluster_centers), "subcluster_centers must be a finite 2D matrix")
@icontract.ensure(lambda result, subcluster_centers: result == np.asarray(subcluster_centers).shape[0], "_n_features_out must equal the number of subcluster centers")
def birch_n_features_out(subcluster_centers: NDArray[np.float64]) -> int:
    """Return Birch's transformed output width from the subcluster-centers matrix."""
    return int(np.asarray(subcluster_centers).shape[0])
