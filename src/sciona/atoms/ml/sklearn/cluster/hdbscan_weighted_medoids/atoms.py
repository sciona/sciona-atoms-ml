"""HDBSCAN weighted-medoid helper atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_hdbscan_medoid,
    witness_hdbscan_medoid_index,
    witness_hdbscan_medoid_weighted_distance_sums,
    witness_hdbscan_medoid_weighted_distances,
)


def _square_distance_matrix(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 2
        and array.shape[0] >= 1
        and array.shape[0] == array.shape[1]
        and np.all(np.isfinite(array))
        and np.all(array >= 0.0)
    )


def _probability_vector_with_length(value: object, length: int) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 1
        and array.shape == (length,)
        and np.all(np.isfinite(array))
        and np.all((array >= 0.0) & (array <= 1.0))
    )


def _weighted_distance_matrix_like(result: object, distance_matrix: object) -> bool:
    values = np.asarray(result, dtype=np.float64)
    source = np.asarray(distance_matrix, dtype=np.float64)
    return bool(values.shape == source.shape and np.all(np.isfinite(values)) and np.all(values >= 0.0))


def _square_weighted_distance_matrix(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 2
        and array.shape[0] >= 1
        and array.shape[0] == array.shape[1]
        and np.all(np.isfinite(array))
        and np.all(array >= 0.0)
    )


def _distance_sum_vector(result: object, weighted_distances: object) -> bool:
    values = np.asarray(result, dtype=np.float64)
    source = np.asarray(weighted_distances, dtype=np.float64)
    return bool(values.shape == (source.shape[0],) and np.all(np.isfinite(values)) and np.all(values >= 0.0))


def _finite_vector(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _medoid_index_valid(index: int, data: object) -> bool:
    matrix = np.asarray(data, dtype=np.float64)
    return bool(isinstance(index, int) and 0 <= index < matrix.shape[0])


def _finite_matrix(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _medoid_vector(result: object, data: object) -> bool:
    values = np.asarray(result, dtype=np.float64)
    matrix = np.asarray(data, dtype=np.float64)
    return bool(values.shape == (matrix.shape[1],) and np.all(np.isfinite(values)))


@register_atom(witness_hdbscan_medoid_weighted_distances)
@icontract.require(lambda distance_matrix: _square_distance_matrix(distance_matrix), "distance_matrix must be a finite square nonnegative matrix")
@icontract.require(
    lambda distance_matrix, strength: _probability_vector_with_length(strength, np.asarray(distance_matrix, dtype=np.float64).shape[0]),
    "strength must be a finite probability vector aligned with distance_matrix",
)
@icontract.ensure(lambda result, distance_matrix: _weighted_distance_matrix_like(result, distance_matrix), "result must be a weighted distance matrix aligned with distance_matrix")
def hdbscan_medoid_weighted_distances(
    distance_matrix: NDArray[np.float64],
    strength: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Broadcast cluster strengths across a pairwise-distance matrix for medoid selection."""
    return np.asarray(distance_matrix, dtype=np.float64) * np.asarray(strength, dtype=np.float64)


@register_atom(witness_hdbscan_medoid_weighted_distance_sums)
@icontract.require(lambda weighted_distances: _square_weighted_distance_matrix(weighted_distances), "weighted_distances must be a finite square nonnegative matrix")
@icontract.ensure(lambda result, weighted_distances: _distance_sum_vector(result, weighted_distances), "result must be one finite nonnegative weighted-distance sum per row")
def hdbscan_medoid_weighted_distance_sums(
    weighted_distances: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Reduce one weighted pairwise-distance matrix to per-row sums."""
    return np.asarray(np.asarray(weighted_distances, dtype=np.float64).sum(axis=1), dtype=np.float64)


@register_atom(witness_hdbscan_medoid_index)
@icontract.require(lambda weighted_distance_sums: _finite_vector(weighted_distance_sums), "weighted_distance_sums must be a nonempty finite vector")
@icontract.ensure(
    lambda result, weighted_distance_sums: isinstance(result, int) and 0 <= result < np.asarray(weighted_distance_sums, dtype=np.float64).shape[0],
    "result must be a valid row index into weighted_distance_sums",
)
def hdbscan_medoid_index(
    weighted_distance_sums: NDArray[np.float64],
) -> int:
    """Select HDBSCAN's medoid row index from weighted pairwise-distance sums."""
    return int(np.argmin(np.asarray(weighted_distance_sums, dtype=np.float64)))


@register_atom(witness_hdbscan_medoid)
@icontract.require(lambda data: _finite_matrix(data), "data must be a finite two-dimensional matrix")
@icontract.require(lambda data, medoid_index: _medoid_index_valid(medoid_index, data), "medoid_index must select an existing row in data")
@icontract.ensure(lambda result, data: _medoid_vector(result, data), "result must be a finite feature vector with one entry per data column")
def hdbscan_medoid(
    data: NDArray[np.float64],
    medoid_index: int,
) -> NDArray[np.float64]:
    """Return the selected HDBSCAN medoid row."""
    return np.asarray(np.asarray(data, dtype=np.float64)[int(medoid_index)], dtype=np.float64)
