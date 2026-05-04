"""Spectral biclustering KMeans shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_bicluster_kmeans_centroids,
    witness_bicluster_kmeans_labels,
    witness_bicluster_project_cluster_labels,
    witness_bicluster_use_minibatch_kmeans,
)


def _bool_value(value: object) -> bool:
    return isinstance(value, bool)


def _finite_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 2
        and array.shape[0] >= 1
        and array.shape[1] >= 1
        and np.all(np.isfinite(array))
    )


def _nonnegative_int_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.int64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(array >= 0))


@register_atom(witness_bicluster_use_minibatch_kmeans)
@icontract.require(lambda mini_batch: _bool_value(mini_batch), "mini_batch must be boolean")
@icontract.ensure(lambda result: _bool_value(result), "branch selection must be boolean")
def bicluster_use_minibatch_kmeans(
    mini_batch: bool,
) -> bool:
    """Resolve whether spectral biclustering dispatches to MiniBatchKMeans."""
    return bool(mini_batch)


@register_atom(witness_bicluster_kmeans_centroids)
@icontract.require(lambda cluster_centers: _finite_matrix(cluster_centers), "cluster_centers must be a finite nonempty matrix")
@icontract.ensure(
    lambda result, cluster_centers: _finite_matrix(result) and np.asarray(result).shape == np.asarray(cluster_centers).shape,
    "centroid matrix must preserve the fitted cluster-center shape",
)
def bicluster_kmeans_centroids(
    cluster_centers: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Expose the centroid matrix from a fitted KMeans or MiniBatchKMeans step."""
    return np.asarray(cluster_centers, dtype=np.float64)


@register_atom(witness_bicluster_kmeans_labels)
@icontract.require(lambda labels: _nonnegative_int_vector(labels), "labels must be a nonempty nonnegative integer vector")
@icontract.ensure(
    lambda result, labels: _nonnegative_int_vector(result) and np.asarray(result).shape == np.asarray(labels).shape,
    "label vector must preserve the fitted KMeans label shape",
)
def bicluster_kmeans_labels(
    labels: NDArray[np.int64],
) -> NDArray[np.int64]:
    """Expose the label vector from a fitted KMeans or MiniBatchKMeans step."""
    return np.asarray(labels, dtype=np.int64)


@register_atom(witness_bicluster_project_cluster_labels)
@icontract.require(lambda labels: _nonnegative_int_vector(labels), "labels must be a nonempty nonnegative integer vector")
@icontract.ensure(
    lambda result, labels: _nonnegative_int_vector(result) and np.asarray(result).shape == np.asarray(labels).shape,
    "project-and-cluster labels must preserve the fitted KMeans label shape",
)
def bicluster_project_cluster_labels(
    labels: NDArray[np.int64],
) -> NDArray[np.int64]:
    """Model SpectralBiclustering._project_and_cluster returning fitted labels after deferred KMeans."""
    return np.asarray(labels, dtype=np.int64)
