"""BIRCH global-clustering shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_birch_global_short_circuit_required,
    witness_birch_not_enough_centroids_warning_message,
    witness_birch_not_enough_centroids_warning_required,
    witness_birch_partial_fit_global_only_required,
)


def _bool_value(value: object) -> bool:
    return isinstance(value, bool)


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


@register_atom(witness_birch_partial_fit_global_only_required)
@icontract.require(lambda has_input_data: _bool_value(has_input_data), "has_input_data must be boolean")
@icontract.ensure(lambda result, has_input_data: result == (not has_input_data), "partial_fit global-only branch must trigger when X is None")
def birch_partial_fit_global_only_required(has_input_data: bool) -> bool:
    """Return whether Birch.partial_fit should run only the global clustering step."""
    return not has_input_data


@register_atom(witness_birch_global_short_circuit_required)
@icontract.require(lambda clusterer_is_none: _bool_value(clusterer_is_none), "clusterer_is_none must be boolean")
@icontract.require(lambda not_enough_centroids: _bool_value(not_enough_centroids), "not_enough_centroids must be boolean")
@icontract.ensure(lambda result, clusterer_is_none, not_enough_centroids: result == (clusterer_is_none or not_enough_centroids), "short-circuit branch must match Birch._global_clustering")
def birch_global_short_circuit_required(
    clusterer_is_none: bool,
    not_enough_centroids: bool,
) -> bool:
    """Return whether Birch should bypass global clustering and keep identity subcluster labels."""
    return bool(clusterer_is_none or not_enough_centroids)


@register_atom(witness_birch_not_enough_centroids_warning_required)
@icontract.require(lambda not_enough_centroids: _bool_value(not_enough_centroids), "not_enough_centroids must be boolean")
@icontract.ensure(lambda result, not_enough_centroids: result == not_enough_centroids, "warning predicate must equal the too-few-subclusters flag")
def birch_not_enough_centroids_warning_required(not_enough_centroids: bool) -> bool:
    """Return whether Birch should emit the too-few-subclusters warning."""
    return bool(not_enough_centroids)


@register_atom(witness_birch_not_enough_centroids_warning_message)
@icontract.require(lambda n_centroids: _positive_int(n_centroids), "n_centroids must be positive")
@icontract.require(lambda n_clusters: _positive_int(n_clusters), "n_clusters must be positive")
@icontract.ensure(
    lambda result, n_centroids, n_clusters: result
    == "Number of subclusters found (%d) by BIRCH is less than (%d). Decrease the threshold."
    % (n_centroids, n_clusters),
    "warning message must match Birch._global_clustering",
)
def birch_not_enough_centroids_warning_message(
    n_centroids: int,
    n_clusters: int,
) -> str:
    """Format Birch's too-few-subclusters warning message."""
    return "Number of subclusters found (%d) by BIRCH is less than (%d). Decrease the threshold." % (
        n_centroids,
        n_clusters,
    )
