"""Ghost witnesses for spectral biclustering KMeans shell atoms."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def witness_bicluster_use_minibatch_kmeans(mini_batch: bool) -> bool:
    """Describe the branch that selects MiniBatchKMeans instead of KMeans."""
    return bool(mini_batch)


def witness_bicluster_kmeans_centroids(
    cluster_centers: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Describe the centroid matrix exposed from a fitted KMeans-like model."""
    return np.asarray(cluster_centers, dtype=np.float64)


def witness_bicluster_kmeans_labels(
    labels: NDArray[np.int64],
) -> NDArray[np.int64]:
    """Describe the labels exposed from a fitted KMeans-like model."""
    return np.asarray(labels, dtype=np.int64)


def witness_bicluster_project_cluster_labels(
    labels: NDArray[np.int64],
) -> NDArray[np.int64]:
    """Describe the labels returned from biclustering's project-and-cluster shell."""
    return np.asarray(labels, dtype=np.int64)
