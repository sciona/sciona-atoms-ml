"""Ghost witnesses for HDBSCAN weighted-medoid helper atoms."""

from __future__ import annotations

from numpy.typing import NDArray


def witness_hdbscan_medoid_weighted_distances(
    distance_matrix: NDArray[float],
    strength: NDArray[float],
) -> NDArray[float]:
    """Describe HDBSCAN's weighted pairwise-distance matrix for medoid selection."""
    del distance_matrix
    del strength
    raise NotImplementedError


def witness_hdbscan_medoid_weighted_distance_sums(
    weighted_distances: NDArray[float],
) -> NDArray[float]:
    """Describe the weighted pairwise-distance row sums used for medoid selection."""
    del weighted_distances
    raise NotImplementedError


def witness_hdbscan_medoid_index(weighted_distance_sums: NDArray[float]) -> int:
    """Describe HDBSCAN's medoid row index from weighted distance sums."""
    del weighted_distance_sums
    return 0


def witness_hdbscan_medoid(data: NDArray[float], medoid_index: int) -> NDArray[float]:
    """Describe the medoid feature row selected from one HDBSCAN cluster."""
    del data
    del medoid_index
    raise NotImplementedError
