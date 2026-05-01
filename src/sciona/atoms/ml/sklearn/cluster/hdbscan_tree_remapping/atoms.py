"""HDBSCAN linkage-tree remapping helper atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Mapping

import icontract
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster._hdbscan.hdbscan import HIERARCHY_dtype

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_hdbscan_outlier_linkage_rows,
    witness_hdbscan_remapped_single_linkage_tree,
    witness_hdbscan_remapped_tree_rows,
    witness_hdbscan_tree_node_id,
)


_TREE_FIELDS = ("left_node", "right_node", "value", "cluster_size")


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _strictly_increasing_indices(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=np.int64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and (array.shape[0] == 0 or np.all(array[1:] > array[:-1])) and np.all(array >= 0))


def _mapping_valid(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    keys = list(value.keys())
    vals = list(value.values())
    return bool(
        all(isinstance(k, int) and k >= 0 for k in keys)
        and set(keys) == set(range(len(keys)))
        and all(isinstance(v, int) and v >= 0 for v in vals)
        and len(set(vals)) == len(vals)
    )


def _hierarchy_tree_valid(value: object) -> bool:
    try:
        tree = np.asarray(value, dtype=HIERARCHY_dtype)
    except (TypeError, ValueError):
        return False
    if tree.ndim != 1 or tree.shape[0] < 1 or tree.dtype.names != _TREE_FIELDS:
        return False
    return bool(
        np.all(tree["left_node"] >= 0)
        and np.all(tree["right_node"] >= 0)
        and np.all(np.isfinite(tree["value"]))
        and np.all(tree["value"] >= 0.0)
        and np.all(tree["cluster_size"] >= 2)
    )


def _node_inputs_valid(node_id: object, finite_count: object, outlier_count: object, internal_to_raw: object) -> bool:
    if not (_nonnegative_int(node_id) and _nonnegative_int(outlier_count) and _mapping_valid(internal_to_raw)):
        return False
    return isinstance(finite_count, int) and finite_count == len(internal_to_raw) and finite_count >= 1


def _node_result_valid(result: object) -> bool:
    return isinstance(result, int) and result >= 0


def _tree_rows_inputs_valid(tree: object, internal_to_raw: object, outlier_count: object) -> bool:
    return bool(_hierarchy_tree_valid(tree) and _mapping_valid(internal_to_raw) and _nonnegative_int(outlier_count))


def _tree_rows_valid(result: object, tree: object) -> bool:
    values = np.asarray(result, dtype=HIERARCHY_dtype)
    source = np.asarray(tree, dtype=HIERARCHY_dtype)
    return bool(values.shape == source.shape and values.dtype.names == _TREE_FIELDS)


def _outlier_rows_valid(result: object, nonfinite_raw_indices: object) -> bool:
    values = np.asarray(result, dtype=HIERARCHY_dtype)
    indices = np.asarray(nonfinite_raw_indices, dtype=np.int64)
    return bool(values.shape == (indices.shape[0],) and values.dtype.names == _TREE_FIELDS)


def _remapped_tree_valid(result: object, tree: object, nonfinite_raw_indices: object) -> bool:
    values = np.asarray(result, dtype=HIERARCHY_dtype)
    source = np.asarray(tree, dtype=HIERARCHY_dtype)
    indices = np.asarray(nonfinite_raw_indices, dtype=np.int64)
    return bool(values.shape == (source.shape[0] + indices.shape[0],) and values.dtype.names == _TREE_FIELDS)


@register_atom(witness_hdbscan_tree_node_id)
@icontract.require(lambda node_id, finite_count, outlier_count, internal_to_raw: _node_inputs_valid(node_id, finite_count, outlier_count, internal_to_raw), "node id, finite count, outlier count, and internal_to_raw must be compatible with HDBSCAN tree remapping")
@icontract.ensure(lambda result: _node_result_valid(result), "remapped node id must be a nonnegative integer")
def hdbscan_tree_node_id(
    node_id: int,
    finite_count: int,
    outlier_count: int,
    internal_to_raw: dict[int, int],
) -> int:
    """Remap one HDBSCAN linkage-tree node id back onto the raw sample axis."""
    if int(node_id) < int(finite_count):
        return int(internal_to_raw[int(node_id)])
    return int(node_id) + int(outlier_count)


@register_atom(witness_hdbscan_remapped_tree_rows)
@icontract.require(lambda tree, internal_to_raw, outlier_count: _tree_rows_inputs_valid(tree, internal_to_raw, outlier_count), "tree, internal_to_raw, and outlier_count must be compatible with HDBSCAN tree remapping")
@icontract.ensure(lambda result, tree: _tree_rows_valid(result, tree), "remapped tree rows must preserve the linkage-tree shape and dtype")
def hdbscan_remapped_tree_rows(
    tree: NDArray[np.void],
    internal_to_raw: dict[int, int],
    outlier_count: int,
) -> NDArray[np.void]:
    """Remap HDBSCAN linkage-tree endpoint ids before appending outlier rows."""
    values = np.asarray(tree, dtype=HIERARCHY_dtype).copy()
    finite_count = len(internal_to_raw)
    for i in range(values.shape[0]):
        values[i]["left_node"] = hdbscan_tree_node_id(
            int(values[i]["left_node"]),
            finite_count,
            int(outlier_count),
            internal_to_raw,
        )
        values[i]["right_node"] = hdbscan_tree_node_id(
            int(values[i]["right_node"]),
            finite_count,
            int(outlier_count),
            internal_to_raw,
        )
    return values


@register_atom(witness_hdbscan_outlier_linkage_rows)
@icontract.require(lambda last_cluster_id: _nonnegative_int(last_cluster_id), "last_cluster_id must be nonnegative")
@icontract.require(lambda last_cluster_size: _positive_int(last_cluster_size), "last_cluster_size must be positive")
@icontract.require(lambda nonfinite_raw_indices: _strictly_increasing_indices(nonfinite_raw_indices), "nonfinite_raw_indices must be a sorted raw-index vector")
@icontract.ensure(lambda result, nonfinite_raw_indices: _outlier_rows_valid(result, nonfinite_raw_indices), "outlier rows must match the non-finite index count and hierarchy dtype")
def hdbscan_outlier_linkage_rows(
    last_cluster_id: int,
    last_cluster_size: int,
    nonfinite_raw_indices: NDArray[np.int64],
) -> NDArray[np.void]:
    """Build HDBSCAN's appended outlier linkage rows at infinite distance."""
    outliers = np.asarray(nonfinite_raw_indices, dtype=np.int64)
    outlier_tree = np.zeros(outliers.shape[0], dtype=HIERARCHY_dtype)
    cluster_id = int(last_cluster_id)
    cluster_size = int(last_cluster_size)
    for i, outlier in enumerate(outliers):
        outlier_tree[i] = (int(outlier), cluster_id + 1, np.inf, cluster_size + 1)
        cluster_id += 1
        cluster_size += 1
    return outlier_tree


@register_atom(witness_hdbscan_remapped_single_linkage_tree)
@icontract.require(lambda tree, internal_to_raw, nonfinite_raw_indices: _hierarchy_tree_valid(tree) and _mapping_valid(internal_to_raw) and _strictly_increasing_indices(nonfinite_raw_indices), "tree, internal_to_raw, and nonfinite_raw_indices must be valid for HDBSCAN linkage-tree remapping")
@icontract.ensure(lambda result, tree, nonfinite_raw_indices: _remapped_tree_valid(result, tree, nonfinite_raw_indices), "remapped linkage tree must append one outlier row per non-finite raw index")
def hdbscan_remapped_single_linkage_tree(
    tree: NDArray[np.void],
    internal_to_raw: dict[int, int],
    nonfinite_raw_indices: NDArray[np.int64],
) -> NDArray[np.void]:
    """Restore non-finite raw samples into HDBSCAN's single-linkage tree."""
    values = hdbscan_remapped_tree_rows(
        np.asarray(tree, dtype=HIERARCHY_dtype),
        internal_to_raw,
        int(np.asarray(nonfinite_raw_indices, dtype=np.int64).shape[0]),
    )
    outliers = np.asarray(nonfinite_raw_indices, dtype=np.int64)
    outlier_rows = hdbscan_outlier_linkage_rows(
        max(int(values[-1]["left_node"]), int(values[-1]["right_node"])),
        int(values[-1]["cluster_size"]),
        outliers,
    )
    return np.concatenate([values, outlier_rows])
