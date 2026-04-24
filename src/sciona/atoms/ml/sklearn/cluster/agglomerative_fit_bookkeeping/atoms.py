"""Agglomerative fit-bookkeeping atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_agglomerative_cluster_count_from_distances,
    witness_agglomerative_labels_from_heads,
    witness_agglomerative_resolve_compute_full_tree,
    witness_agglomerative_resolve_tree_n_clusters,
    witness_agglomerative_return_distance_required,
)


def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _compute_full_tree_spec_valid(value: str | bool) -> bool:
    return bool(isinstance(value, bool) or value == "auto")


def _distance_threshold_valid(value: float | None) -> bool:
    return bool(value is None or (isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value))))


def _distances_valid(distances: NDArray[np.float64]) -> bool:
    values = np.asarray(distances, dtype=np.float64)
    return bool(values.ndim == 1 and values.shape[0] >= 1 and np.all(np.isfinite(values)))


def _heads_valid(heads: NDArray[np.int64], n_samples: int) -> bool:
    values = np.asarray(heads)
    return bool(
        _positive_int(n_samples)
        and values.ndim == 1
        and values.shape[0] >= n_samples
        and np.issubdtype(values.dtype, np.integer)
    )


def _labels_valid(result: NDArray[np.int64], heads: NDArray[np.int64], n_samples: int) -> bool:
    labels = np.asarray(result)
    head_values = np.asarray(heads)
    sample_heads = np.asarray(head_values[:n_samples], dtype=np.int64)
    unique_heads = np.unique(sample_heads)
    relabeled = np.searchsorted(unique_heads, sample_heads)
    return bool(
        labels.shape == (n_samples,)
        and np.issubdtype(labels.dtype, np.integer)
        and np.array_equal(labels, relabeled)
    )


@register_atom(witness_agglomerative_resolve_compute_full_tree)
@icontract.require(lambda compute_full_tree: _compute_full_tree_spec_valid(compute_full_tree), "compute_full_tree must be a bool or 'auto'")
@icontract.require(lambda n_clusters: _positive_int(n_clusters), "n_clusters must be a positive integer")
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be a positive integer")
@icontract.require(lambda distance_threshold: _distance_threshold_valid(distance_threshold), "distance_threshold must be finite or None")
@icontract.ensure(lambda result: isinstance(result, bool), "resolved compute_full_tree flag must be boolean")
def agglomerative_resolve_compute_full_tree(
    compute_full_tree: str | bool,
    has_connectivity: bool,
    n_clusters: int,
    n_samples: int,
    *,
    distance_threshold: float | None = None,
) -> bool:
    """Resolve sklearn's effective compute_full_tree flag before tree construction."""
    if not has_connectivity:
        return True
    if compute_full_tree == "auto":
        if distance_threshold is not None:
            return True
        return bool(n_clusters < max(100, 0.02 * n_samples))
    return bool(compute_full_tree)


@register_atom(witness_agglomerative_resolve_tree_n_clusters)
@icontract.require(lambda n_clusters: _positive_int(n_clusters), "n_clusters must be a positive integer")
@icontract.ensure(lambda result, n_clusters, compute_full_tree: result is None if compute_full_tree else result == n_clusters, "tree n_clusters must be None for full trees and otherwise preserve n_clusters")
def agglomerative_resolve_tree_n_clusters(
    n_clusters: int,
    compute_full_tree: bool,
) -> int | None:
    """Resolve the n_clusters argument passed to sklearn's tree builder."""
    return None if compute_full_tree else int(n_clusters)


@register_atom(witness_agglomerative_return_distance_required)
@icontract.require(lambda distance_threshold: _distance_threshold_valid(distance_threshold), "distance_threshold must be finite or None")
@icontract.ensure(lambda result: isinstance(result, bool), "return_distance flag must be boolean")
def agglomerative_return_distance_required(
    *,
    distance_threshold: float | None = None,
    compute_distances: bool = False,
) -> bool:
    """Decide whether agglomerative tree construction must also return merge distances."""
    return bool(distance_threshold is not None or compute_distances)


@register_atom(witness_agglomerative_cluster_count_from_distances)
@icontract.require(lambda distances: _distances_valid(distances), "distances must be a nonempty finite 1D array")
@icontract.require(lambda distance_threshold: isinstance(distance_threshold, (int, float)) and not isinstance(distance_threshold, bool) and np.isfinite(float(distance_threshold)), "distance_threshold must be finite")
@icontract.ensure(lambda result: _positive_int(result), "derived cluster count must be a positive integer")
def agglomerative_cluster_count_from_distances(
    distances: NDArray[np.float64],
    *,
    distance_threshold: float,
) -> int:
    """Derive sklearn's n_clusters_ from merge distances and a distance threshold."""
    values = np.asarray(distances, dtype=np.float64)
    return int(np.count_nonzero(values >= float(distance_threshold)) + 1)


@register_atom(witness_agglomerative_labels_from_heads)
@icontract.require(lambda heads, n_samples: _heads_valid(heads, n_samples), "heads must be an integer vector covering the sample prefix")
@icontract.ensure(lambda result, heads, n_samples: _labels_valid(result, heads, n_samples), "labels must be the reindexed unique heads for the sample prefix")
def agglomerative_labels_from_heads(
    heads: NDArray[np.int64],
    *,
    n_samples: int,
) -> NDArray[np.int64]:
    """Relabel agglomerative parent-head ids on the sample prefix into compact cluster labels."""
    sample_heads = np.asarray(heads[:n_samples], dtype=np.int64)
    unique_heads = np.unique(sample_heads)
    return np.asarray(np.searchsorted(unique_heads, sample_heads), dtype=np.int64)
