"""DBSCAN fit-bookkeeping atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_dbscan_core_sample_mask,
    witness_dbscan_dense_core_components,
    witness_dbscan_empty_components,
    witness_dbscan_initial_noise_labels,
    witness_dbscan_neighbor_count_vector,
    witness_dbscan_precomputed_sparse_self_neighbors,
    witness_dbscan_weighted_neighbor_sums,
)

NeighborhoodTuple = tuple[NDArray[np.intp], ...]


def _nonempty_square_sparse(X: object) -> bool:
    return bool(
        sp.issparse(X)
        and X.ndim == 2
        and X.shape[0] >= 1
        and X.shape[0] == X.shape[1]
        and np.all(np.isfinite(X.data))
        and np.all(X.data >= 0.0)
    )


def _csr_self_neighbor_result_valid(result: object, X: sp.spmatrix) -> bool:
    if not sp.isspmatrix_csr(result):
        return False
    diagonal = np.asarray(X.diagonal(), dtype=np.float64)
    result_diagonal = np.asarray(result.diagonal(), dtype=np.float64)
    return bool(
        result.shape == X.shape
        and np.all(np.isfinite(result.data))
        and np.array_equal(result_diagonal, diagonal)
    )


def _neighborhoods_valid(neighborhoods: object) -> bool:
    if not isinstance(neighborhoods, tuple) or len(neighborhoods) < 1:
        return False
    for block in neighborhoods:
        try:
            values = np.asarray(block, dtype=np.intp)
        except (TypeError, ValueError):
            return False
        if values.ndim != 1 or np.any(values < 0):
            return False
    return True


def _neighbor_count_result_valid(result: object, neighborhoods: NeighborhoodTuple) -> bool:
    values = np.asarray(result)
    expected = np.asarray([len(block) for block in neighborhoods], dtype=np.int64)
    return bool(values.shape == expected.shape and np.issubdtype(values.dtype, np.integer) and np.array_equal(values, expected))


def _sample_weight_valid(sample_weight: object, n_samples: int) -> bool:
    try:
        values = np.asarray(sample_weight, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 1 and values.shape[0] == n_samples and np.all(np.isfinite(values)))


def _weighted_neighbor_result_valid(result: object, neighborhoods: NeighborhoodTuple, sample_weight: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    expected = np.asarray([np.sum(sample_weight[block]) for block in neighborhoods], dtype=np.float64)
    return bool(values.shape == expected.shape and np.all(np.isfinite(values)) and np.allclose(values, expected))


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _neighbor_mass_valid(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _core_mask_result_valid(result: object, neighbor_mass: object, min_samples: int) -> bool:
    values = np.asarray(result)
    mass = np.asarray(neighbor_mass, dtype=np.float64)
    expected = np.asarray(mass >= min_samples, dtype=np.uint8)
    return bool(values.shape == expected.shape and values.dtype == np.uint8 and np.array_equal(values, expected))


def _noise_labels_valid(result: object, n_samples: int) -> bool:
    values = np.asarray(result)
    return bool(values.shape == (n_samples,) and np.issubdtype(values.dtype, np.integer) and np.all(values == -1))


def _dense_matrix_valid(X: object) -> bool:
    try:
        values = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 2 and values.shape[0] >= 1 and values.shape[1] >= 1 and np.all(np.isfinite(values)))


def _core_indices_valid(core_sample_indices: object, n_samples: int) -> bool:
    try:
        values = np.asarray(core_sample_indices, dtype=np.intp)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 1 and np.all(values >= 0) and np.all(values < n_samples))


def _core_components_valid(result: object, X: object, core_sample_indices: object) -> bool:
    values = np.asarray(result, dtype=np.float64)
    source = np.asarray(X, dtype=np.float64)
    indices = np.asarray(core_sample_indices, dtype=np.intp)
    expected = source[indices].copy()
    return bool(values.shape == expected.shape and np.allclose(values, expected))


def _empty_components_valid(result: object, n_features: int) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(values.shape == (0, n_features))


@register_atom(witness_dbscan_precomputed_sparse_self_neighbors)
@icontract.require(lambda X: _nonempty_square_sparse(X), "X must be a finite nonnegative sparse square matrix")
@icontract.ensure(lambda result, X: _csr_self_neighbor_result_valid(result, X), "result must be a CSR sparse matrix with explicit diagonal self-neighbor entries")
def dbscan_precomputed_sparse_self_neighbors(X: sp.spmatrix) -> sp.csr_matrix:
    """Copy a sparse precomputed distance graph and make the diagonal explicit."""
    graph = X.tocsr(copy=True)
    graph.setdiag(graph.diagonal())
    return graph


@register_atom(witness_dbscan_neighbor_count_vector)
@icontract.require(lambda neighborhoods: _neighborhoods_valid(neighborhoods), "neighborhoods must be a nonempty tuple of 1D nonnegative index vectors")
@icontract.ensure(lambda result, neighborhoods: _neighbor_count_result_valid(result, neighborhoods), "neighbor counts must equal the length of each neighborhood vector")
def dbscan_neighbor_count_vector(neighborhoods: NeighborhoodTuple) -> NDArray[np.int64]:
    """Count the number of neighbor indices for each sample."""
    return np.asarray([len(block) for block in neighborhoods], dtype=np.int64)


@register_atom(witness_dbscan_weighted_neighbor_sums)
@icontract.require(lambda neighborhoods: _neighborhoods_valid(neighborhoods), "neighborhoods must be a nonempty tuple of 1D nonnegative index vectors")
@icontract.require(lambda neighborhoods, sample_weight: _sample_weight_valid(sample_weight, len(neighborhoods)), "sample_weight must be a finite vector aligned with the neighborhoods")
@icontract.ensure(lambda result, neighborhoods, sample_weight: _weighted_neighbor_result_valid(result, neighborhoods, sample_weight), "weighted neighbor sums must equal the sample-weight mass of each neighborhood")
def dbscan_weighted_neighbor_sums(
    neighborhoods: NeighborhoodTuple,
    sample_weight: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Sum sample weights over each DBSCAN neighborhood."""
    weights = np.asarray(sample_weight, dtype=np.float64)
    return np.asarray([np.sum(weights[block]) for block in neighborhoods], dtype=np.float64)


@register_atom(witness_dbscan_core_sample_mask)
@icontract.require(lambda neighbor_mass: _neighbor_mass_valid(neighbor_mass), "neighbor_mass must be a finite 1D vector")
@icontract.require(lambda min_samples: _positive_int(min_samples), "min_samples must be positive")
@icontract.ensure(lambda result, neighbor_mass, min_samples: _core_mask_result_valid(result, neighbor_mass, min_samples), "core mask must be the uint8 threshold test against min_samples")
def dbscan_core_sample_mask(
    neighbor_mass: NDArray[np.float64] | NDArray[np.int64],
    min_samples: int,
) -> NDArray[np.uint8]:
    """Threshold neighborhood mass into DBSCAN's uint8 core-sample mask."""
    return np.asarray(np.asarray(neighbor_mass, dtype=np.float64) >= min_samples, dtype=np.uint8)


@register_atom(witness_dbscan_initial_noise_labels)
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be positive")
@icontract.ensure(lambda result, n_samples: _noise_labels_valid(result, n_samples), "labels must initialize to all noise (-1)")
def dbscan_initial_noise_labels(n_samples: int) -> NDArray[np.intp]:
    """Initialize DBSCAN labels to the all-noise vector."""
    return np.full(n_samples, -1, dtype=np.intp)


@register_atom(witness_dbscan_dense_core_components)
@icontract.require(lambda X: _dense_matrix_valid(X), "X must be a finite dense 2D matrix")
@icontract.require(lambda X, core_sample_indices: _core_indices_valid(core_sample_indices, np.asarray(X).shape[0]), "core_sample_indices must be a valid 1D index vector into X")
@icontract.ensure(lambda result, X, core_sample_indices: _core_components_valid(result, X, core_sample_indices), "components must be the copied dense core-sample rows")
def dbscan_dense_core_components(
    X: NDArray[np.float64],
    core_sample_indices: NDArray[np.intp],
) -> NDArray[np.float64]:
    """Copy the dense core-sample rows selected by DBSCAN."""
    values = np.asarray(X, dtype=np.float64)
    indices = np.asarray(core_sample_indices, dtype=np.intp)
    return np.asarray(values[indices].copy(), dtype=np.float64)


@register_atom(witness_dbscan_empty_components)
@icontract.require(lambda n_features: _positive_int(n_features), "n_features must be positive")
@icontract.ensure(lambda result, n_features: _empty_components_valid(result, n_features), "empty components must have zero rows and the requested feature count")
def dbscan_empty_components(n_features: int) -> NDArray[np.float64]:
    """Return DBSCAN's empty dense components fallback."""
    return np.empty((0, n_features), dtype=np.float64)
