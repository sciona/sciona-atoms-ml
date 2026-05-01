"""Ghost witnesses for BIRCH global-clustering shell atoms."""

from __future__ import annotations


def witness_birch_partial_fit_global_only_required(has_input_data: bool) -> bool:
    """Describe the partial_fit global-only branch predicate."""
    return bool(has_input_data)


def witness_birch_global_short_circuit_required(
    clusterer_is_none: bool,
    not_enough_centroids: bool,
) -> bool:
    """Describe the no-global-clustering branch predicate."""
    return bool(clusterer_is_none or not_enough_centroids)


def witness_birch_not_enough_centroids_warning_required(not_enough_centroids: bool) -> bool:
    """Describe whether Birch should emit its too-few-subclusters warning."""
    return bool(not_enough_centroids)


def witness_birch_not_enough_centroids_warning_message(
    n_centroids: int,
    n_clusters: int,
) -> str:
    """Describe Birch's too-few-subclusters warning text."""
    if n_centroids < 1:
        raise ValueError("n_centroids must be positive")
    if n_clusters < 1:
        raise ValueError("n_clusters must be positive")
    return ""
