"""HDBSCAN fit-setup helper atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
from sklearn.cluster._hdbscan.hdbscan import FAST_METRICS
from sklearn.neighbors import BallTree, KDTree

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_hdbscan_backend_leaf_size,
    witness_hdbscan_backend_name,
    witness_hdbscan_backend_uses_copy,
    witness_hdbscan_require_min_samples_within_sample_count,
    witness_hdbscan_require_multiple_samples,
    witness_hdbscan_resolved_min_samples,
    witness_hdbscan_sparse_forced_algorithm_guard,
    witness_hdbscan_store_centers_precomputed_guard,
    witness_hdbscan_tree_metric_compatibility_guard,
)


_VALID_ALGORITHMS = {"auto", "brute", "kd_tree", "ball_tree"}
_BACKEND_NAMES = {"brute", "kd_tree", "ball_tree"}


def _string_value(value: object) -> bool:
    return isinstance(value, str) and len(value) >= 1


def _bool_value(value: object) -> bool:
    return isinstance(value, bool)


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _algorithm_value(value: object) -> bool:
    return isinstance(value, str) and value in _VALID_ALGORITHMS


def _backend_name_value(value: object) -> bool:
    return isinstance(value, str) and value in _BACKEND_NAMES


def _optional_leaf_size_valid(result: object, backend_name: str, leaf_size: int) -> bool:
    if backend_name == "brute":
        return result is None
    return result == int(leaf_size)


@register_atom(witness_hdbscan_store_centers_precomputed_guard)
@icontract.require(lambda metric: _string_value(metric), "metric must be a nonempty string")
@icontract.ensure(lambda result: _bool_value(result), "guard result must be boolean")
def hdbscan_store_centers_precomputed_guard(
    metric: str,
    store_centers: object,
) -> bool:
    """Apply HDBSCAN's guard against store_centers with precomputed distances."""
    if metric == "precomputed" and store_centers is not None:
        raise ValueError("Cannot store centers when using a precomputed distance matrix.")
    return True


@register_atom(witness_hdbscan_resolved_min_samples)
@icontract.require(lambda min_cluster_size: _positive_int(min_cluster_size), "min_cluster_size must be positive")
@icontract.require(lambda min_samples: min_samples is None or _positive_int(min_samples), "min_samples must be positive or None")
@icontract.ensure(lambda result: _positive_int(result), "resolved min_samples must be positive")
def hdbscan_resolved_min_samples(
    min_cluster_size: int,
    min_samples: int | None,
) -> int:
    """Resolve HDBSCAN's internal _min_samples value."""
    return int(min_cluster_size if min_samples is None else min_samples)


@register_atom(witness_hdbscan_require_multiple_samples)
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be positive")
@icontract.ensure(lambda result: _bool_value(result), "guard result must be boolean")
def hdbscan_require_multiple_samples(
    n_samples: int,
) -> bool:
    """Apply HDBSCAN's requirement that fit input contain more than one sample."""
    if int(n_samples) == 1:
        raise ValueError("n_samples=1 while HDBSCAN requires more than one sample")
    return True


@register_atom(witness_hdbscan_require_min_samples_within_sample_count)
@icontract.require(lambda resolved_min_samples: _positive_int(resolved_min_samples), "resolved_min_samples must be positive")
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be positive")
@icontract.ensure(lambda result: _bool_value(result), "guard result must be boolean")
def hdbscan_require_min_samples_within_sample_count(
    resolved_min_samples: int,
    n_samples: int,
) -> bool:
    """Apply HDBSCAN's resolved min_samples upper-bound guard."""
    if int(resolved_min_samples) > int(n_samples):
        raise ValueError(
            f"min_samples ({int(resolved_min_samples)}) must be at most the number of samples in X ({int(n_samples)})"
        )
    return True


@register_atom(witness_hdbscan_tree_metric_compatibility_guard)
@icontract.require(lambda algorithm: _algorithm_value(algorithm), "algorithm must be one of HDBSCAN's supported algorithm strings")
@icontract.require(lambda metric: _string_value(metric), "metric must be a nonempty string")
@icontract.ensure(lambda result: _bool_value(result), "guard result must be boolean")
def hdbscan_tree_metric_compatibility_guard(
    algorithm: str,
    metric: str,
) -> bool:
    """Apply HDBSCAN's explicit KDTree and BallTree metric compatibility guard."""
    if algorithm == "kd_tree" and metric not in KDTree.valid_metrics:
        raise ValueError(
            f"{metric} is not a valid metric for a KDTree-based algorithm. Please select a different metric."
        )
    if algorithm == "ball_tree" and metric not in BallTree.valid_metrics:
        raise ValueError(
            f"{metric} is not a valid metric for a BallTree-based algorithm. Please select a different metric."
        )
    return True


@register_atom(witness_hdbscan_sparse_forced_algorithm_guard)
@icontract.require(lambda metric: _string_value(metric), "metric must be a nonempty string")
@icontract.require(lambda is_sparse: _bool_value(is_sparse), "is_sparse must be boolean")
@icontract.require(lambda algorithm: _algorithm_value(algorithm), "algorithm must be one of HDBSCAN's supported algorithm strings")
@icontract.ensure(lambda result: _bool_value(result), "guard result must be boolean")
def hdbscan_sparse_forced_algorithm_guard(
    metric: str,
    is_sparse: bool,
    algorithm: str,
) -> bool:
    """Apply HDBSCAN's sparse-data guard for explicit non-auto algorithms."""
    if metric != "precomputed" and bool(is_sparse) and algorithm != "auto" and algorithm != "brute":
        raise ValueError("Sparse data matrices only support algorithm `brute`.")
    return True


@register_atom(witness_hdbscan_backend_name)
@icontract.require(lambda metric: _string_value(metric), "metric must be a nonempty string")
@icontract.require(lambda is_sparse: _bool_value(is_sparse), "is_sparse must be boolean")
@icontract.require(lambda algorithm: _algorithm_value(algorithm), "algorithm must be one of HDBSCAN's supported algorithm strings")
@icontract.ensure(lambda result: _backend_name_value(result), "backend name must be brute, kd_tree, or ball_tree")
def hdbscan_backend_name(
    metric: str,
    is_sparse: bool,
    algorithm: str,
) -> str:
    """Resolve HDBSCAN's MST backend name before native execution."""
    if algorithm != "auto":
        if algorithm == "brute":
            return "brute"
        if algorithm == "kd_tree":
            return "kd_tree"
        return "ball_tree"

    if bool(is_sparse) or metric not in FAST_METRICS:
        return "brute"
    if metric in KDTree.valid_metrics:
        return "kd_tree"
    return "ball_tree"


@register_atom(witness_hdbscan_backend_uses_copy)
@icontract.require(lambda backend_name: _backend_name_value(backend_name), "backend_name must be brute, kd_tree, or ball_tree")
@icontract.ensure(lambda result: _bool_value(result), "copy-usage flag must be boolean")
def hdbscan_backend_uses_copy(
    backend_name: str,
) -> bool:
    """Return whether the selected HDBSCAN backend passes the copy kwarg."""
    return backend_name == "brute"


@register_atom(witness_hdbscan_backend_leaf_size)
@icontract.require(lambda backend_name: _backend_name_value(backend_name), "backend_name must be brute, kd_tree, or ball_tree")
@icontract.require(lambda leaf_size: _positive_int(leaf_size), "leaf_size must be positive")
@icontract.ensure(lambda result, backend_name, leaf_size: _optional_leaf_size_valid(result, backend_name, leaf_size), "leaf_size must be omitted for brute and preserved for tree backends")
def hdbscan_backend_leaf_size(
    backend_name: str,
    leaf_size: int,
) -> int | None:
    """Return the HDBSCAN leaf_size kwarg for the selected backend."""
    if backend_name == "brute":
        return None
    return int(leaf_size)
