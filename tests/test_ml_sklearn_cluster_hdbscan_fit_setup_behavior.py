from __future__ import annotations

import pytest


def test_hdbscan_fit_setup_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.hdbscan_fit_setup import (
        hdbscan_backend_leaf_size,
        hdbscan_backend_name,
        hdbscan_backend_uses_copy,
        hdbscan_require_min_samples_within_sample_count,
        hdbscan_require_multiple_samples,
        hdbscan_resolved_min_samples,
        hdbscan_sparse_forced_algorithm_guard,
        hdbscan_store_centers_precomputed_guard,
        hdbscan_tree_metric_compatibility_guard,
    )

    assert callable(hdbscan_store_centers_precomputed_guard)
    assert callable(hdbscan_resolved_min_samples)
    assert callable(hdbscan_require_multiple_samples)
    assert callable(hdbscan_require_min_samples_within_sample_count)
    assert callable(hdbscan_tree_metric_compatibility_guard)
    assert callable(hdbscan_sparse_forced_algorithm_guard)
    assert callable(hdbscan_backend_name)
    assert callable(hdbscan_backend_uses_copy)
    assert callable(hdbscan_backend_leaf_size)


def test_hdbscan_basic_fit_setup_helpers_match_source_logic() -> None:
    from sciona.atoms.ml.sklearn.cluster.hdbscan_fit_setup import (
        hdbscan_require_min_samples_within_sample_count,
        hdbscan_require_multiple_samples,
        hdbscan_resolved_min_samples,
        hdbscan_store_centers_precomputed_guard,
    )

    assert hdbscan_store_centers_precomputed_guard("euclidean", None) is True
    assert hdbscan_resolved_min_samples(5, None) == 5
    assert hdbscan_resolved_min_samples(5, 3) == 3
    assert hdbscan_require_multiple_samples(2) is True
    assert hdbscan_require_min_samples_within_sample_count(3, 3) is True


def test_hdbscan_backend_selection_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.cluster.hdbscan_fit_setup import (
        hdbscan_backend_leaf_size,
        hdbscan_backend_name,
        hdbscan_backend_uses_copy,
    )

    assert hdbscan_backend_name("euclidean", False, "brute") == "brute"
    assert hdbscan_backend_uses_copy("brute") is True
    assert hdbscan_backend_leaf_size("brute", 40) is None

    assert hdbscan_backend_name("euclidean", False, "kd_tree") == "kd_tree"
    assert hdbscan_backend_uses_copy("kd_tree") is False
    assert hdbscan_backend_leaf_size("kd_tree", 40) == 40

    assert hdbscan_backend_name("braycurtis", False, "ball_tree") == "ball_tree"
    assert hdbscan_backend_leaf_size("ball_tree", 24) == 24

    assert hdbscan_backend_name("euclidean", True, "auto") == "brute"
    assert hdbscan_backend_name("cosine", False, "auto") == "brute"
    assert hdbscan_backend_name("euclidean", False, "auto") == "kd_tree"
    assert hdbscan_backend_name("braycurtis", False, "auto") == "ball_tree"


def test_hdbscan_fit_setup_guards_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.cluster.hdbscan_fit_setup import (
        hdbscan_require_min_samples_within_sample_count,
        hdbscan_require_multiple_samples,
        hdbscan_sparse_forced_algorithm_guard,
        hdbscan_store_centers_precomputed_guard,
        hdbscan_tree_metric_compatibility_guard,
    )

    with pytest.raises(ValueError, match="Cannot store centers"):
        hdbscan_store_centers_precomputed_guard("precomputed", "medoid")

    with pytest.raises(ValueError, match="requires more than one sample"):
        hdbscan_require_multiple_samples(1)

    with pytest.raises(ValueError, match="must be at most the number of samples"):
        hdbscan_require_min_samples_within_sample_count(4, 3)

    with pytest.raises(ValueError, match="KDTree-based algorithm"):
        hdbscan_tree_metric_compatibility_guard("kd_tree", "braycurtis")

    with pytest.raises(ValueError, match="BallTree-based algorithm"):
        hdbscan_tree_metric_compatibility_guard("ball_tree", "cosine")

    with pytest.raises(ValueError, match="Sparse data matrices only support algorithm `brute`"):
        hdbscan_sparse_forced_algorithm_guard("euclidean", True, "kd_tree")
