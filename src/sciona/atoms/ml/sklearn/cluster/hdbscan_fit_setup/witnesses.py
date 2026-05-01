"""Ghost witnesses for HDBSCAN fit-setup helper atoms."""

from __future__ import annotations


def witness_hdbscan_store_centers_precomputed_guard(
    metric: str,
    store_centers: object,
) -> bool:
    """Describe HDBSCAN's precomputed-distance guard for store_centers."""
    del metric
    del store_centers
    return True


def witness_hdbscan_resolved_min_samples(
    min_cluster_size: int,
    min_samples: int | None,
) -> int:
    """Describe HDBSCAN's resolved internal min_samples value."""
    del min_cluster_size
    del min_samples
    return 1


def witness_hdbscan_require_multiple_samples(
    n_samples: int,
) -> bool:
    """Describe HDBSCAN's requirement that fit input contain more than one sample."""
    del n_samples
    return True


def witness_hdbscan_require_min_samples_within_sample_count(
    resolved_min_samples: int,
    n_samples: int,
) -> bool:
    """Describe HDBSCAN's guard that resolved min_samples not exceed sample count."""
    del resolved_min_samples
    del n_samples
    return True


def witness_hdbscan_tree_metric_compatibility_guard(
    algorithm: str,
    metric: str,
) -> bool:
    """Describe HDBSCAN's KDTree and BallTree metric compatibility guard."""
    del algorithm
    del metric
    return True


def witness_hdbscan_sparse_forced_algorithm_guard(
    metric: str,
    is_sparse: bool,
    algorithm: str,
) -> bool:
    """Describe HDBSCAN's sparse-data guard for explicit non-auto algorithms."""
    del metric
    del is_sparse
    del algorithm
    return True


def witness_hdbscan_backend_name(
    metric: str,
    is_sparse: bool,
    algorithm: str,
) -> str:
    """Describe HDBSCAN's MST backend selection before native execution."""
    del metric
    del is_sparse
    del algorithm
    return "brute"


def witness_hdbscan_backend_uses_copy(
    backend_name: str,
) -> bool:
    """Describe whether the selected HDBSCAN backend passes the copy kwarg."""
    del backend_name
    return False


def witness_hdbscan_backend_leaf_size(
    backend_name: str,
    leaf_size: int,
) -> int | None:
    """Describe whether the selected HDBSCAN backend passes leaf_size."""
    del backend_name
    del leaf_size
    return None
