"""Ghost witnesses for sklearn Birch output-math helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_birch_subcluster_norms(subcluster_centers: AbstractArray) -> AbstractArray:
    """Describe Birch's cached squared norms for subcluster centers."""
    if len(subcluster_centers.shape) != 2:
        raise ValueError("subcluster_centers must be 2D")
    if int(subcluster_centers.shape[0]) < 1 or int(subcluster_centers.shape[1]) < 1:
        raise ValueError("subcluster_centers must be nonempty")
    return AbstractArray(shape=(int(subcluster_centers.shape[0]),), dtype="float64")


def witness_birch_predict_argmin(
    X: AbstractArray,
    subcluster_centers: AbstractArray,
    subcluster_norms: AbstractArray,
) -> AbstractArray:
    """Describe Birch's nearest-subcluster assignment indices."""
    if len(X.shape) != 2 or len(subcluster_centers.shape) != 2:
        raise ValueError("X and subcluster_centers must be 2D")
    if int(X.shape[0]) < 1 or int(X.shape[1]) < 1:
        raise ValueError("X must be nonempty")
    if int(subcluster_centers.shape[0]) < 1 or int(subcluster_centers.shape[1]) < 1:
        raise ValueError("subcluster_centers must be nonempty")
    if int(X.shape[1]) != int(subcluster_centers.shape[1]):
        raise ValueError("X and subcluster_centers must share a feature count")
    if len(subcluster_norms.shape) != 1 or int(subcluster_norms.shape[0]) != int(subcluster_centers.shape[0]):
        raise ValueError("subcluster_norms must match the number of centers")
    return AbstractArray(shape=(int(X.shape[0]),), dtype="int64")


def witness_birch_predict_labels(
    nearest_subcluster_indices: AbstractArray,
    subcluster_labels: AbstractArray,
) -> AbstractArray:
    """Describe Birch's label lookup from nearest subcluster indices."""
    if len(nearest_subcluster_indices.shape) != 1 or len(subcluster_labels.shape) != 1:
        raise ValueError("nearest_subcluster_indices and subcluster_labels must be 1D")
    return AbstractArray(shape=(int(nearest_subcluster_indices.shape[0]),), dtype="int64")


def witness_birch_transform_distances(
    X: AbstractArray,
    subcluster_centers: AbstractArray,
) -> AbstractArray:
    """Describe Birch's transform distance matrix."""
    if len(X.shape) != 2 or len(subcluster_centers.shape) != 2:
        raise ValueError("X and subcluster_centers must be 2D")
    if int(X.shape[0]) < 1 or int(X.shape[1]) < 1:
        raise ValueError("X must be nonempty")
    if int(subcluster_centers.shape[0]) < 1 or int(subcluster_centers.shape[1]) < 1:
        raise ValueError("subcluster_centers must be nonempty")
    if int(X.shape[1]) != int(subcluster_centers.shape[1]):
        raise ValueError("X and subcluster_centers must share a feature count")
    return AbstractArray(
        shape=(int(X.shape[0]), int(subcluster_centers.shape[0])),
        dtype="float64",
    )
