"""Agglomerative hierarchy cut helper atoms adapted from scikit-learn."""

from __future__ import annotations

from heapq import heappush, heappushpop

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_agglomerative_descendent_leaves,
    witness_agglomerative_hc_cut,
    witness_agglomerative_root_node,
)


def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _children_matrix_valid(children: NDArray[np.int64], n_leaves: int) -> bool:
    if not _positive_int(n_leaves):
        return False
    try:
        values = np.asarray(children, dtype=np.int64)
    except (TypeError, ValueError):
        return False
    if values.ndim != 2 or values.shape != (n_leaves - 1, 2):
        return False
    if n_leaves == 1:
        return True
    max_node = 2 * n_leaves - 2
    if np.any(values < 0) or np.any(values > max_node):
        return False
    for merge_index, pair in enumerate(values):
        current_node = n_leaves + merge_index
        if np.any(pair >= current_node):
            return False
    return True


def _node_valid(node: int, children: NDArray[np.int64], n_leaves: int) -> bool:
    return bool(_children_matrix_valid(children, n_leaves) and isinstance(node, int) and not isinstance(node, bool) and 0 <= node <= 2 * n_leaves - 2)


def _cluster_count_valid(n_clusters: int, children: NDArray[np.int64], n_leaves: int) -> bool:
    return bool(_children_matrix_valid(children, n_leaves) and _positive_int(n_clusters) and n_clusters <= n_leaves)


def _root_valid(result: int, n_leaves: int) -> bool:
    return bool(isinstance(result, (int, np.integer)) and int(result) == 2 * n_leaves - 2)


def _descendent_result_valid(result: NDArray[np.int64], n_leaves: int) -> bool:
    values = np.asarray(result)
    return bool(
        values.ndim == 1
        and np.issubdtype(values.dtype, np.integer)
        and values.shape[0] >= 1
        and np.all(values >= 0)
        and np.all(values < n_leaves)
        and np.unique(values).shape[0] == values.shape[0]
    )


def _cut_result_valid(result: NDArray[np.int64], n_clusters: int, n_leaves: int) -> bool:
    labels = np.asarray(result)
    return bool(
        labels.shape == (n_leaves,)
        and np.issubdtype(labels.dtype, np.integer)
        and np.all(labels >= 0)
        and np.all(labels < n_clusters)
        and np.unique(labels).shape[0] == n_clusters
    )


@register_atom(witness_agglomerative_root_node)
@icontract.require(lambda children, n_leaves: _children_matrix_valid(children, n_leaves), "children must encode a valid binary hierarchy")
@icontract.ensure(lambda result, n_leaves: _root_valid(result, n_leaves), "root node must be the final hierarchy node")
def agglomerative_root_node(children: NDArray[np.int64], n_leaves: int) -> int:
    """Return the root node id of a sklearn agglomerative children array."""
    del children
    return int(2 * n_leaves - 2)


@register_atom(witness_agglomerative_descendent_leaves)
@icontract.require(lambda node, children, n_leaves: _node_valid(node, children, n_leaves), "node must be valid for the hierarchy")
@icontract.ensure(lambda result, n_leaves: _descendent_result_valid(result, n_leaves), "descendants must be unique leaf ids")
def agglomerative_descendent_leaves(
    node: int,
    children: NDArray[np.int64],
    n_leaves: int,
) -> NDArray[np.int64]:
    """Return leaf descendants for a node in a sklearn agglomerative tree."""
    values = np.asarray(children, dtype=np.int64)
    if node < n_leaves:
        return np.asarray([node], dtype=np.int64)

    leaves: list[int] = []
    stack = [int(node)]
    while stack:
        current = stack.pop()
        if current < n_leaves:
            leaves.append(current)
        else:
            left, right = values[current - n_leaves]
            stack.append(int(right))
            stack.append(int(left))
    return np.asarray(leaves, dtype=np.int64)


@register_atom(witness_agglomerative_hc_cut)
@icontract.require(lambda n_clusters, children, n_leaves: _cluster_count_valid(n_clusters, children, n_leaves), "cluster count and hierarchy must be compatible")
@icontract.ensure(lambda result, n_clusters, n_leaves: _cut_result_valid(result, n_clusters, n_leaves), "cut labels must cover each requested cluster")
def agglomerative_hc_cut(
    n_clusters: int,
    children: NDArray[np.int64],
    n_leaves: int,
) -> NDArray[np.int64]:
    """Cut a sklearn agglomerative hierarchy into cluster labels."""
    values = np.asarray(children, dtype=np.int64)
    if n_leaves == 1:
        return np.zeros(1, dtype=np.int64)

    nodes = [-(max(values[-1]) + 1)]
    for _ in range(n_clusters - 1):
        these_children = values[-nodes[0] - n_leaves]
        heappush(nodes, -int(these_children[0]))
        heappushpop(nodes, -int(these_children[1]))

    labels = np.zeros(n_leaves, dtype=np.int64)
    for label, node in enumerate(nodes):
        labels[agglomerative_descendent_leaves(int(-node), values, n_leaves)] = label
    return labels
