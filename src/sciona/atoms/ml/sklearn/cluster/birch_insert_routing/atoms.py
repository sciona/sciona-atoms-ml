"""BIRCH insert-routing helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_birch_insert_append_with_split_required,
    witness_birch_insert_append_without_split_required,
    witness_birch_insert_child_split_required,
    witness_birch_insert_child_update_required,
    witness_birch_insert_closest_index,
    witness_birch_insert_closest_scores,
    witness_birch_insert_parent_split_required,
)


def _finite_centroid_matrix(value: object) -> bool:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(matrix.ndim == 2 and matrix.shape[0] >= 1 and matrix.shape[1] >= 1 and np.all(np.isfinite(matrix)))


def _finite_vector(value: object, *, min_len: int = 1) -> bool:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(vector.ndim == 1 and vector.shape[0] >= min_len and np.all(np.isfinite(vector)))


def _squared_norm_valid(value: object, centroids: NDArray[np.float64]) -> bool:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(vector.ndim == 1 and vector.shape[0] == np.asarray(centroids).shape[0] and np.all(np.isfinite(vector)))


def _candidate_width_matches(candidate_centroid: NDArray[np.float64], centroids: NDArray[np.float64]) -> bool:
    return bool(np.asarray(candidate_centroid).shape[0] == np.asarray(centroids).shape[1])


def _score_vector_valid(value: object, n_clusters: int) -> bool:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(vector.ndim == 1 and vector.shape[0] == n_clusters and np.all(np.isfinite(vector)))


def _positive_int(value: object, *, minimum: int = 1) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _bool_value(value: object) -> bool:
    return isinstance(value, bool)


@register_atom(witness_birch_insert_closest_scores)
@icontract.require(lambda centroids: _finite_centroid_matrix(centroids), "centroids must be a nonempty finite 2D matrix")
@icontract.require(lambda squared_norm, centroids: _squared_norm_valid(squared_norm, centroids), "squared_norm must be a finite vector matching centroid count")
@icontract.require(lambda candidate_centroid: _finite_vector(candidate_centroid), "candidate_centroid must be a finite 1D vector")
@icontract.require(lambda candidate_centroid, centroids: _candidate_width_matches(candidate_centroid, centroids), "candidate_centroid must match centroid width")
@icontract.ensure(lambda result, centroids: _score_vector_valid(result, np.asarray(centroids).shape[0]), "result must be a finite score vector matching centroid count")
def birch_insert_closest_scores(
    centroids: NDArray[np.float64],
    squared_norm: NDArray[np.float64],
    candidate_centroid: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute sklearn's closest-subcluster score vector for BIRCH insertion."""
    score_values = np.dot(np.asarray(centroids, dtype=np.float64), np.asarray(candidate_centroid, dtype=np.float64))
    score_values *= -2.0
    score_values += np.asarray(squared_norm, dtype=np.float64)
    return np.asarray(score_values, dtype=np.float64)


@register_atom(witness_birch_insert_closest_index)
@icontract.require(lambda closest_scores: _finite_vector(closest_scores), "closest_scores must be a finite nonempty 1D vector")
@icontract.ensure(lambda result, closest_scores: _positive_int(result, minimum=0) and result < np.asarray(closest_scores).shape[0], "result must be a valid score index")
def birch_insert_closest_index(closest_scores: NDArray[np.float64]) -> int:
    """Select sklearn's closest subcluster index for BIRCH insertion."""
    return int(np.argmin(np.asarray(closest_scores, dtype=np.float64)))


@register_atom(witness_birch_insert_child_update_required)
@icontract.require(lambda has_child: _bool_value(has_child), "has_child must be boolean")
@icontract.require(lambda split_child: _bool_value(split_child), "split_child must be boolean")
@icontract.ensure(lambda result, has_child, split_child: result == (has_child and not split_child), "result must match the child-update branch predicate")
def birch_insert_child_update_required(has_child: bool, split_child: bool) -> bool:
    """Return whether BIRCH insertion follows the recursive child-update branch."""
    return bool(has_child and not split_child)


@register_atom(witness_birch_insert_child_split_required)
@icontract.require(lambda has_child: _bool_value(has_child), "has_child must be boolean")
@icontract.require(lambda split_child: _bool_value(split_child), "split_child must be boolean")
@icontract.ensure(lambda result, has_child, split_child: result == (has_child and split_child), "result must match the child-split branch predicate")
def birch_insert_child_split_required(has_child: bool, split_child: bool) -> bool:
    """Return whether BIRCH insertion follows the recursive child-split branch."""
    return bool(has_child and split_child)


@register_atom(witness_birch_insert_append_without_split_required)
@icontract.require(lambda has_child: _bool_value(has_child), "has_child must be boolean")
@icontract.require(lambda merged: _bool_value(merged), "merged must be boolean")
@icontract.require(lambda current_count: _positive_int(current_count), "current_count must be positive")
@icontract.require(lambda branching_factor: _positive_int(branching_factor, minimum=2), "branching_factor must be at least two")
@icontract.ensure(lambda result, has_child, merged, current_count, branching_factor: result == ((not has_child) and (not merged) and current_count < branching_factor), "result must match the append-without-split predicate")
def birch_insert_append_without_split_required(
    has_child: bool,
    merged: bool,
    current_count: int,
    branching_factor: int,
) -> bool:
    """Return whether BIRCH insertion appends a new subcluster without splitting."""
    return bool((not has_child) and (not merged) and current_count < branching_factor)


@register_atom(witness_birch_insert_append_with_split_required)
@icontract.require(lambda has_child: _bool_value(has_child), "has_child must be boolean")
@icontract.require(lambda merged: _bool_value(merged), "merged must be boolean")
@icontract.require(lambda current_count: _positive_int(current_count), "current_count must be positive")
@icontract.require(lambda branching_factor: _positive_int(branching_factor, minimum=2), "branching_factor must be at least two")
@icontract.ensure(lambda result, has_child, merged, current_count, branching_factor: result == ((not has_child) and (not merged) and current_count >= branching_factor), "result must match the append-and-split predicate")
def birch_insert_append_with_split_required(
    has_child: bool,
    merged: bool,
    current_count: int,
    branching_factor: int,
) -> bool:
    """Return whether BIRCH insertion appends a new subcluster and signals overflow."""
    return bool((not has_child) and (not merged) and current_count >= branching_factor)


@register_atom(witness_birch_insert_parent_split_required)
@icontract.require(lambda updated_count: _positive_int(updated_count), "updated_count must be positive")
@icontract.require(lambda branching_factor: _positive_int(branching_factor, minimum=2), "branching_factor must be at least two")
@icontract.ensure(lambda result, updated_count, branching_factor: result == (updated_count > branching_factor), "result must match the parent-overflow predicate")
def birch_insert_parent_split_required(updated_count: int, branching_factor: int) -> bool:
    """Return whether a parent node must split after accommodating recursive child partitions."""
    return bool(updated_count > branching_factor)
