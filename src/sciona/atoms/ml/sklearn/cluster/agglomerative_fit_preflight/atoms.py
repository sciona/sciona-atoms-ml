"""Agglomerative fit preflight atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_agglomerative_fit_require_exactly_one_cluster_spec,
    witness_agglomerative_fit_require_full_tree_when_distance_threshold_set,
    witness_agglomerative_fit_require_ward_metric_euclidean,
)


def _positive_int_or_none(value: int | None) -> bool:
    return bool(
        value is None
        or (isinstance(value, int) and not isinstance(value, bool) and value >= 1)
    )


def _distance_threshold_valid(value: float | None) -> bool:
    return bool(
        value is None
        or (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and np.isfinite(float(value))
        )
    )


def _compute_full_tree_spec_valid(value: str | bool) -> bool:
    return bool(isinstance(value, bool) or value == "auto")


@register_atom(witness_agglomerative_fit_require_exactly_one_cluster_spec)
@icontract.require(
    lambda n_clusters: _positive_int_or_none(n_clusters),
    "n_clusters must be a positive integer or None",
)
@icontract.require(
    lambda distance_threshold: _distance_threshold_valid(distance_threshold),
    "distance_threshold must be finite or None",
)
@icontract.require(
    lambda n_clusters, distance_threshold: (n_clusters is None) ^ (distance_threshold is None),
    "exactly one of n_clusters and distance_threshold must be set",
)
@icontract.ensure(lambda result: result is True, "successful preflight returns True")
def agglomerative_fit_require_exactly_one_cluster_spec(
    n_clusters: int | None,
    *,
    distance_threshold: float | None = None,
) -> bool:
    """Require sklearn's exactly-one-of n_clusters or distance_threshold preflight rule."""
    return True


@register_atom(witness_agglomerative_fit_require_full_tree_when_distance_threshold_set)
@icontract.require(
    lambda compute_full_tree: _compute_full_tree_spec_valid(compute_full_tree),
    "compute_full_tree must be a bool or 'auto'",
)
@icontract.require(
    lambda distance_threshold: _distance_threshold_valid(distance_threshold),
    "distance_threshold must be finite or None",
)
@icontract.require(
    lambda compute_full_tree, distance_threshold: distance_threshold is None or bool(compute_full_tree),
    "compute_full_tree must be truthy when distance_threshold is set",
)
@icontract.ensure(lambda result: result is True, "successful preflight returns True")
def agglomerative_fit_require_full_tree_when_distance_threshold_set(
    compute_full_tree: str | bool,
    *,
    distance_threshold: float | None = None,
) -> bool:
    """Require sklearn's full-tree preflight rule for distance-threshold mode."""
    return True


@register_atom(witness_agglomerative_fit_require_ward_metric_euclidean)
@icontract.require(lambda linkage: isinstance(linkage, str) and len(linkage) >= 1, "linkage must be a nonempty string")
@icontract.require(
    lambda linkage, metric: linkage != "ward" or metric == "euclidean",
    "ward linkage requires the literal metric 'euclidean'",
)
@icontract.ensure(lambda result: result is True, "successful preflight returns True")
def agglomerative_fit_require_ward_metric_euclidean(
    linkage: str,
    metric: object,
) -> bool:
    """Require sklearn's Ward-only-euclidean metric preflight rule."""
    return True
