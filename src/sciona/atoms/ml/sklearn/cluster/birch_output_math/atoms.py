"""Birch output-math atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_birch_predict_argmin,
    witness_birch_predict_labels,
    witness_birch_subcluster_norms,
    witness_birch_transform_distances,
)

MatrixLike = NDArray[np.float64] | list[list[float]]
VectorLike = NDArray[np.float64] | list[float]
IndexLike = NDArray[np.int64] | list[int]

def _finite_2d_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))

def _shared_feature_count(X: object, subcluster_centers: object) -> bool:
    return bool(_finite_2d_matrix(X) and _finite_2d_matrix(subcluster_centers) and np.asarray(X).shape[1] == np.asarray(subcluster_centers).shape[1])

def _subcluster_norms_valid(subcluster_norms: object, subcluster_centers: object) -> bool:
    try:
        norms = np.asarray(subcluster_norms, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        _finite_2d_matrix(subcluster_centers)
        and norms.ndim == 1
        and norms.shape[0] == np.asarray(subcluster_centers).shape[0]
        and np.all(np.isfinite(norms))
        and np.all(norms >= 0.0)
    )

def _argmin_valid(result: object, X: object, subcluster_centers: object) -> bool:
    values = np.asarray(result)
    n_samples = np.asarray(X).shape[0]
    n_centers = np.asarray(subcluster_centers).shape[0]
    return bool(
        values.shape == (n_samples,)
        and np.issubdtype(values.dtype, np.integer)
        and np.all(values >= 0)
        and np.all(values < n_centers)
    )

def _subcluster_labels_valid(subcluster_labels: object, subcluster_centers: object) -> bool:
    values = np.asarray(subcluster_labels)
    return bool(
        _finite_2d_matrix(subcluster_centers)
        and values.ndim == 1
        and values.shape[0] == np.asarray(subcluster_centers).shape[0]
        and np.issubdtype(values.dtype, np.integer)
        and np.all(values >= 0)
    )

def _predicted_labels_valid(result: object, nearest_subcluster_indices: object, subcluster_labels: object) -> bool:
    values = np.asarray(result)
    nearest = np.asarray(nearest_subcluster_indices)
    labels = np.asarray(subcluster_labels)
    return bool(
        values.shape == nearest.shape
        and values.ndim == 1
        and np.issubdtype(values.dtype, np.integer)
        and np.all(values >= 0)
        and np.all(np.isin(values, labels))
    )

def _distance_matrix_valid(result: object, X: object, subcluster_centers: object) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(
        values.shape == (np.asarray(X).shape[0], np.asarray(subcluster_centers).shape[0])
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
    )

@register_atom(witness_birch_subcluster_norms)
@icontract.require(lambda subcluster_centers: _finite_2d_matrix(subcluster_centers), "subcluster_centers must be a finite nonempty 2D matrix")
@icontract.ensure(lambda result, subcluster_centers: _subcluster_norms_valid(result, subcluster_centers), "squared norms must match the number of centers")
def birch_subcluster_norms(subcluster_centers: MatrixLike) -> NDArray[np.float64]:
    from sklearn.utils.extmath import row_norms
    """Return Birch's cached squared subcluster norms from subcluster centers."""
    centers = np.asarray(subcluster_centers, dtype=np.float64)
    return np.asarray(row_norms(centers, squared=True), dtype=np.float64)

@register_atom(witness_birch_predict_argmin)
@icontract.require(lambda X: _finite_2d_matrix(X), "X must be a finite nonempty 2D matrix")
@icontract.require(lambda subcluster_centers: _finite_2d_matrix(subcluster_centers), "subcluster_centers must be a finite nonempty 2D matrix")
@icontract.require(lambda X, subcluster_centers: _shared_feature_count(X, subcluster_centers), "X and subcluster_centers must share a feature count")
@icontract.require(lambda subcluster_norms, subcluster_centers: _subcluster_norms_valid(subcluster_norms, subcluster_centers), "subcluster_norms must match the number of centers")
@icontract.ensure(lambda result, X, subcluster_centers: _argmin_valid(result, X, subcluster_centers), "nearest-center indices must match Birch predict argmins")
def birch_predict_argmin(
    X: MatrixLike,
    subcluster_centers: MatrixLike,
    subcluster_norms: VectorLike,
) -> NDArray[np.int64]:
    from sklearn.metrics import pairwise_distances_argmin
    """Return Birch's nearest-subcluster indices from supplied centers and squared norms."""
    X_array = np.asarray(X, dtype=np.float64)
    centers = np.asarray(subcluster_centers, dtype=np.float64)
    norms = np.asarray(subcluster_norms, dtype=np.float64)
    return np.asarray(
        pairwise_distances_argmin(X_array, centers, metric_kwargs={"Y_norm_squared": norms}),
        dtype=np.int64,
    )

@register_atom(witness_birch_predict_labels)
@icontract.require(lambda nearest_subcluster_indices: np.asarray(nearest_subcluster_indices).ndim == 1, "nearest_subcluster_indices must be 1D")
@icontract.require(lambda subcluster_labels, nearest_subcluster_indices: _subcluster_labels_valid(subcluster_labels, np.zeros((np.asarray(subcluster_labels).shape[0], 1), dtype=np.float64)) and np.all(np.asarray(nearest_subcluster_indices) >= 0) and np.all(np.asarray(nearest_subcluster_indices) < np.asarray(subcluster_labels).shape[0]), "indices must address the supplied subcluster_labels vector")
@icontract.ensure(lambda result, nearest_subcluster_indices, subcluster_labels: _predicted_labels_valid(result, nearest_subcluster_indices, subcluster_labels), "predicted labels must be selected from subcluster_labels")
def birch_predict_labels(
    nearest_subcluster_indices: IndexLike,
    subcluster_labels: IndexLike,
) -> NDArray[np.int64]:
    """Return Birch's predicted labels from nearest-subcluster indices and subcluster labels."""
    indices = np.asarray(nearest_subcluster_indices, dtype=np.int64)
    labels = np.asarray(subcluster_labels, dtype=np.int64)
    return np.asarray(labels[indices], dtype=np.int64)

@register_atom(witness_birch_transform_distances)
@icontract.require(lambda X: _finite_2d_matrix(X), "X must be a finite nonempty 2D matrix")
@icontract.require(lambda subcluster_centers: _finite_2d_matrix(subcluster_centers), "subcluster_centers must be a finite nonempty 2D matrix")
@icontract.require(lambda X, subcluster_centers: _shared_feature_count(X, subcluster_centers), "X and subcluster_centers must share a feature count")
@icontract.ensure(lambda result, X, subcluster_centers: _distance_matrix_valid(result, X, subcluster_centers), "distance matrix must match Birch transform output shape")
def birch_transform_distances(
    X: MatrixLike,
    subcluster_centers: MatrixLike,
) -> NDArray[np.float64]:
    from sklearn.metrics.pairwise import euclidean_distances
    """Return Birch's transform distance matrix from samples to subcluster centers."""
    X_array = np.asarray(X, dtype=np.float64)
    centers = np.asarray(subcluster_centers, dtype=np.float64)
    return np.asarray(euclidean_distances(X_array, centers), dtype=np.float64)
