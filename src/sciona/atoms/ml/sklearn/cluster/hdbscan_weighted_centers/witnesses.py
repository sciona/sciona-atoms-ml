"""Ghost witnesses for HDBSCAN weighted-center helper atoms."""

from __future__ import annotations

from numpy.typing import NDArray


def witness_hdbscan_center_cluster_count(labels: NDArray[int]) -> int:
    """Describe HDBSCAN's non-noise cluster count for weighted center computation."""
    del labels
    return 0


def witness_hdbscan_make_centroids(store_centers: object) -> bool:
    """Describe whether HDBSCAN should materialize weighted centroids."""
    del store_centers
    return False


def witness_hdbscan_make_medoids(store_centers: object) -> bool:
    """Describe whether HDBSCAN should materialize weighted medoids."""
    del store_centers
    return False


def witness_hdbscan_center_mask(labels: NDArray[int], cluster_label: int) -> NDArray[bool]:
    """Describe HDBSCAN's per-cluster membership mask."""
    del labels
    del cluster_label
    raise NotImplementedError


def witness_hdbscan_center_data(X: NDArray[float], cluster_mask: NDArray[bool]) -> NDArray[float]:
    """Describe the feature rows selected for one HDBSCAN cluster center."""
    del X
    del cluster_mask
    raise NotImplementedError


def witness_hdbscan_center_strength(probabilities: NDArray[float], cluster_mask: NDArray[bool]) -> NDArray[float]:
    """Describe the membership-strength weights selected for one HDBSCAN cluster center."""
    del probabilities
    del cluster_mask
    raise NotImplementedError


def witness_hdbscan_centroid(data: NDArray[float], strength: NDArray[float]) -> NDArray[float]:
    """Describe HDBSCAN's weighted centroid for one cluster."""
    del data
    del strength
    raise NotImplementedError
