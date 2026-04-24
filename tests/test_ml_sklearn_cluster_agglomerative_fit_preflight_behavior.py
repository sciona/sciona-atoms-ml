from __future__ import annotations

import pytest
from icontract import ViolationError
from sklearn.cluster import AgglomerativeClustering


def test_agglomerative_fit_preflight_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.agglomerative_fit_preflight import (
        agglomerative_fit_require_exactly_one_cluster_spec,
        agglomerative_fit_require_full_tree_when_distance_threshold_set,
        agglomerative_fit_require_ward_metric_euclidean,
    )

    assert callable(agglomerative_fit_require_exactly_one_cluster_spec)
    assert callable(agglomerative_fit_require_full_tree_when_distance_threshold_set)
    assert callable(agglomerative_fit_require_ward_metric_euclidean)


def test_agglomerative_fit_require_exactly_one_cluster_spec_matches_sklearn_valid_cases() -> None:
    from sciona.atoms.ml.sklearn.cluster.agglomerative_fit_preflight import (
        agglomerative_fit_require_exactly_one_cluster_spec,
    )

    assert agglomerative_fit_require_exactly_one_cluster_spec(3, distance_threshold=None) is True
    assert agglomerative_fit_require_exactly_one_cluster_spec(None, distance_threshold=1.5) is True


def test_agglomerative_fit_require_full_tree_when_distance_threshold_set_matches_sklearn_valid_cases() -> None:
    from sciona.atoms.ml.sklearn.cluster.agglomerative_fit_preflight import (
        agglomerative_fit_require_full_tree_when_distance_threshold_set,
    )

    assert agglomerative_fit_require_full_tree_when_distance_threshold_set("auto", distance_threshold=1.5) is True
    assert agglomerative_fit_require_full_tree_when_distance_threshold_set(True, distance_threshold=1.5) is True
    assert agglomerative_fit_require_full_tree_when_distance_threshold_set(False, distance_threshold=None) is True


def test_agglomerative_fit_require_ward_metric_euclidean_matches_sklearn_valid_cases() -> None:
    from sciona.atoms.ml.sklearn.cluster.agglomerative_fit_preflight import (
        agglomerative_fit_require_ward_metric_euclidean,
    )

    assert agglomerative_fit_require_ward_metric_euclidean("ward", "euclidean") is True
    assert agglomerative_fit_require_ward_metric_euclidean("average", "manhattan") is True
    assert agglomerative_fit_require_ward_metric_euclidean("complete", object()) is True


def test_agglomerative_fit_preflight_invalid_cases_match_sklearn_raises() -> None:
    from sciona.atoms.ml.sklearn.cluster.agglomerative_fit_preflight import (
        agglomerative_fit_require_exactly_one_cluster_spec,
        agglomerative_fit_require_full_tree_when_distance_threshold_set,
        agglomerative_fit_require_ward_metric_euclidean,
    )

    with pytest.raises(ValueError, match="Exactly one of n_clusters and distance_threshold"):
        AgglomerativeClustering(n_clusters=2, distance_threshold=1.0).fit([[0.0], [1.0]])
    with pytest.raises(ViolationError):
        agglomerative_fit_require_exactly_one_cluster_spec(2, distance_threshold=1.0)

    with pytest.raises(ValueError, match="Exactly one of n_clusters and distance_threshold"):
        AgglomerativeClustering(n_clusters=None, distance_threshold=None).fit([[0.0], [1.0]])
    with pytest.raises(ViolationError):
        agglomerative_fit_require_exactly_one_cluster_spec(None, distance_threshold=None)

    with pytest.raises(ValueError, match="compute_full_tree must be True if distance_threshold is set"):
        AgglomerativeClustering(n_clusters=None, distance_threshold=1.0, compute_full_tree=False).fit([[0.0], [1.0]])
    with pytest.raises(ViolationError):
        agglomerative_fit_require_full_tree_when_distance_threshold_set(False, distance_threshold=1.0)

    with pytest.raises(ValueError, match="Ward can only work with euclidean distances"):
        AgglomerativeClustering(n_clusters=2, linkage="ward", metric="l2").fit([[0.0], [1.0]])
    with pytest.raises(ViolationError):
        agglomerative_fit_require_ward_metric_euclidean("ward", "l2")


def test_contracts_reject_invalid_agglomerative_fit_preflight_inputs() -> None:
    from sciona.atoms.ml.sklearn.cluster.agglomerative_fit_preflight import (
        agglomerative_fit_require_exactly_one_cluster_spec,
        agglomerative_fit_require_full_tree_when_distance_threshold_set,
    )

    with pytest.raises(ViolationError):
        agglomerative_fit_require_exactly_one_cluster_spec(0, distance_threshold=None)

    with pytest.raises(ViolationError):
        agglomerative_fit_require_full_tree_when_distance_threshold_set("maybe", distance_threshold=None)
