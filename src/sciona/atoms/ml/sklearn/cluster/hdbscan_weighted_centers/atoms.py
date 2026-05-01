"""HDBSCAN weighted-center helper atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_hdbscan_center_cluster_count,
    witness_hdbscan_center_data,
    witness_hdbscan_center_mask,
    witness_hdbscan_center_strength,
    witness_hdbscan_centroid,
    witness_hdbscan_make_centroids,
    witness_hdbscan_make_medoids,
)


_VALID_STORE_CENTERS = {None, "centroid", "medoid", "both"}
_IGNORED_NOISE_LABELS = {-1, -2}


def _integer_vector(value: object) -> bool:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.issubdtype(array.dtype, np.integer))


def _finite_matrix(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _probability_vector(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 1
        and array.shape[0] >= 1
        and np.all(np.isfinite(array))
        and np.all((array >= 0.0) & (array <= 1.0))
    )


def _store_centers_valid(value: object) -> bool:
    return value in _VALID_STORE_CENTERS


def _same_sample_count(X: object, labels: object) -> bool:
    return np.asarray(X, dtype=np.float64).shape[0] == np.asarray(labels).shape[0]


def _mask_like(result: object, labels: object) -> bool:
    values = np.asarray(result)
    source = np.asarray(labels)
    return bool(values.shape == source.shape and values.dtype == np.bool_)


def _mask_valid(mask: object, n_samples: int) -> bool:
    values = np.asarray(mask)
    return bool(values.shape == (n_samples,) and values.dtype == np.bool_ and np.any(values))


def _row_subset_like(result: object, X: object, mask: object) -> bool:
    values = np.asarray(result, dtype=np.float64)
    matrix = np.asarray(X, dtype=np.float64)
    mask_array = np.asarray(mask, dtype=np.bool_)
    return bool(
        values.ndim == 2
        and values.shape == (int(mask_array.sum()), matrix.shape[1])
        and np.all(np.isfinite(values))
    )


def _vector_subset_like(result: object, probabilities: object, mask: object) -> bool:
    values = np.asarray(result, dtype=np.float64)
    source = np.asarray(probabilities, dtype=np.float64)
    mask_array = np.asarray(mask, dtype=np.bool_)
    return bool(
        values.ndim == 1
        and values.shape == (int(mask_array.sum()),)
        and np.all(np.isfinite(values))
        and np.all((values >= 0.0) & (values <= 1.0))
        and source.ndim == 1
    )


def _centroid_inputs_valid(data: object, strength: object) -> bool:
    matrix = np.asarray(data, dtype=np.float64)
    weights = np.asarray(strength, dtype=np.float64)
    return bool(
        matrix.ndim == 2
        and matrix.shape[0] >= 1
        and matrix.shape[1] >= 1
        and weights.ndim == 1
        and weights.shape[0] == matrix.shape[0]
        and np.all(np.isfinite(matrix))
        and np.all(np.isfinite(weights))
        and np.all((weights >= 0.0) & (weights <= 1.0))
        and float(weights.sum()) > 0.0
    )


def _centroid_valid(result: object, data: object) -> bool:
    values = np.asarray(result, dtype=np.float64)
    matrix = np.asarray(data, dtype=np.float64)
    return bool(values.shape == (matrix.shape[1],) and np.all(np.isfinite(values)))


@register_atom(witness_hdbscan_center_cluster_count)
@icontract.require(lambda labels: _integer_vector(labels), "labels must be a one-dimensional integer vector")
@icontract.ensure(lambda result: isinstance(result, int) and result >= 0, "cluster count must be a nonnegative integer")
def hdbscan_center_cluster_count(
    labels: NDArray[np.int_],
) -> int:
    """Count HDBSCAN's non-noise clusters for weighted center computation."""
    return len(set(int(value) for value in np.asarray(labels)) - _IGNORED_NOISE_LABELS)


@register_atom(witness_hdbscan_make_centroids)
@icontract.require(lambda store_centers: _store_centers_valid(store_centers), "store_centers must be None, 'centroid', 'medoid', or 'both'")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def hdbscan_make_centroids(store_centers: object) -> bool:
    """Decide whether HDBSCAN should materialize weighted centroids."""
    return store_centers in ("centroid", "both")


@register_atom(witness_hdbscan_make_medoids)
@icontract.require(lambda store_centers: _store_centers_valid(store_centers), "store_centers must be None, 'centroid', 'medoid', or 'both'")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def hdbscan_make_medoids(store_centers: object) -> bool:
    """Decide whether HDBSCAN should materialize weighted medoids."""
    return store_centers in ("medoid", "both")


@register_atom(witness_hdbscan_center_mask)
@icontract.require(lambda labels: _integer_vector(labels), "labels must be a one-dimensional integer vector")
@icontract.require(lambda cluster_label: isinstance(cluster_label, int) and cluster_label >= 0, "cluster_label must be a nonnegative integer")
@icontract.ensure(lambda result, labels: _mask_like(result, labels), "result must be a boolean mask aligned with labels")
def hdbscan_center_mask(
    labels: NDArray[np.int_],
    cluster_label: int,
) -> NDArray[np.bool_]:
    """Build HDBSCAN's per-cluster mask from fitted labels."""
    return np.asarray(np.asarray(labels) == int(cluster_label), dtype=np.bool_)


@register_atom(witness_hdbscan_center_data)
@icontract.require(lambda X: _finite_matrix(X), "X must be a finite two-dimensional float matrix")
@icontract.require(lambda X, cluster_mask: _mask_valid(cluster_mask, np.asarray(X, dtype=np.float64).shape[0]), "cluster_mask must be a nonempty boolean vector aligned with X")
@icontract.ensure(lambda result, X, cluster_mask: _row_subset_like(result, X, cluster_mask), "result must contain the selected X rows")
def hdbscan_center_data(
    X: NDArray[np.float64],
    cluster_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Extract one HDBSCAN cluster's feature rows for center computation."""
    return np.asarray(X, dtype=np.float64)[np.asarray(cluster_mask, dtype=np.bool_)]


@register_atom(witness_hdbscan_center_strength)
@icontract.require(lambda probabilities: _probability_vector(probabilities), "probabilities must be a one-dimensional finite probability vector")
@icontract.require(lambda probabilities, cluster_mask: _mask_valid(cluster_mask, np.asarray(probabilities, dtype=np.float64).shape[0]), "cluster_mask must be a nonempty boolean vector aligned with probabilities")
@icontract.ensure(lambda result, probabilities, cluster_mask: _vector_subset_like(result, probabilities, cluster_mask), "result must contain the selected probability weights")
def hdbscan_center_strength(
    probabilities: NDArray[np.float64],
    cluster_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Extract one HDBSCAN cluster's membership strengths for center computation."""
    return np.asarray(probabilities, dtype=np.float64)[np.asarray(cluster_mask, dtype=np.bool_)]


@register_atom(witness_hdbscan_centroid)
@icontract.require(lambda data, strength: _centroid_inputs_valid(data, strength), "data and strength must define a nonempty weighted cluster with positive total weight")
@icontract.ensure(lambda result, data: _centroid_valid(result, data), "result must be a finite centroid vector with one entry per feature")
def hdbscan_centroid(
    data: NDArray[np.float64],
    strength: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute HDBSCAN's weighted centroid for one cluster."""
    return np.asarray(np.average(np.asarray(data, dtype=np.float64), weights=np.asarray(strength, dtype=np.float64), axis=0), dtype=np.float64)
