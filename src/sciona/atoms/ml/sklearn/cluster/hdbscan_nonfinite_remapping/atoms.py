"""HDBSCAN non-finite remapping helper atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Mapping

import icontract
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_hdbscan_finite_row_indices,
    witness_hdbscan_infinite_indices,
    witness_hdbscan_internal_to_raw_map,
    witness_hdbscan_missing_indices,
    witness_hdbscan_nonfinite_raw_indices,
    witness_hdbscan_remapped_labels,
    witness_hdbscan_remapped_probabilities,
)

_HDBSCAN_INFINITE_LABEL = -2
_HDBSCAN_INFINITE_PROB = 0.0
_HDBSCAN_MISSING_LABEL = -3

def _finite_row_sums_valid(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1)

def _sample_matrix_valid(value: object) -> bool:
    if sp.issparse(value):
        return bool(value.ndim == 2 and value.shape[0] >= 1 and value.shape[1] >= 1)
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1)

def _index_vector_valid(value: object, n_samples: int) -> bool:
    try:
        array = np.asarray(value, dtype=np.intp)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and np.all(array >= 0) and np.all(array < n_samples))

def _strictly_increasing_indices(value: object, n_samples: int) -> bool:
    if not _index_vector_valid(value, n_samples):
        return False
    array = np.asarray(value, dtype=np.intp)
    return bool(array.shape[0] == 0 or np.all(array[1:] > array[:-1]))

def _mapping_valid(result: object, finite_index: object) -> bool:
    if not isinstance(result, Mapping):
        return False
    indices = np.asarray(finite_index, dtype=np.intp)
    return bool(
        set(result.keys()) == set(range(indices.shape[0]))
        and set(result.values()) == set(int(x) for x in indices)
    )

def _disjoint_index_sets(infinite_index: object, missing_index: object) -> bool:
    infinite = np.asarray(infinite_index, dtype=np.intp)
    missing = np.asarray(missing_index, dtype=np.intp)
    return len(set(int(x) for x in infinite).intersection(int(x) for x in missing)) == 0

def _label_vector_valid(result: object, raw_sample_count: int) -> bool:
    values = np.asarray(result)
    return bool(values.shape == (raw_sample_count,) and np.issubdtype(values.dtype, np.integer))

def _probability_vector_valid(result: object, raw_sample_count: int) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(
        values.shape == (raw_sample_count,)
        and np.all(np.isfinite(values) | np.isnan(values))
        and np.all((values[np.isfinite(values)] >= 0.0) & (values[np.isfinite(values)] <= 1.0))
    )

def _remap_inputs_valid(
    raw_sample_count: object,
    finite_index: object,
    finite_values: object,
    infinite_index: object,
    missing_index: object,
) -> bool:
    if not (isinstance(raw_sample_count, int) and raw_sample_count >= 1):
        return False
    if not (
        _strictly_increasing_indices(finite_index, raw_sample_count)
        and _strictly_increasing_indices(infinite_index, raw_sample_count)
        and _strictly_increasing_indices(missing_index, raw_sample_count)
        and _disjoint_index_sets(infinite_index, missing_index)
    ):
        return False
    finite = np.asarray(finite_index, dtype=np.intp)
    infinite = np.asarray(infinite_index, dtype=np.intp)
    missing = np.asarray(missing_index, dtype=np.intp)
    occupied = set(int(x) for x in finite).union(int(x) for x in infinite).union(int(x) for x in missing)
    if len(occupied) > raw_sample_count:
        return False
    return True

@register_atom(witness_hdbscan_missing_indices)
@icontract.require(lambda reduced_row_sums: _finite_row_sums_valid(reduced_row_sums), "reduced_row_sums must be a 1D float vector")
@icontract.ensure(lambda result, reduced_row_sums: _strictly_increasing_indices(result, np.asarray(reduced_row_sums, dtype=np.float64).shape[0]), "missing indices must be a sorted raw-index vector")
def hdbscan_missing_indices(
    reduced_row_sums: NDArray[np.float64],
) -> NDArray[np.intp]:
    """Extract HDBSCAN's missing-row indices from reduced row sums."""
    values = np.asarray(reduced_row_sums, dtype=np.float64)
    return np.asarray(np.isnan(values).nonzero()[0], dtype=np.intp)

@register_atom(witness_hdbscan_infinite_indices)
@icontract.require(lambda reduced_row_sums: _finite_row_sums_valid(reduced_row_sums), "reduced_row_sums must be a 1D float vector")
@icontract.ensure(lambda result, reduced_row_sums: _strictly_increasing_indices(result, np.asarray(reduced_row_sums, dtype=np.float64).shape[0]), "infinite indices must be a sorted raw-index vector")
def hdbscan_infinite_indices(
    reduced_row_sums: NDArray[np.float64],
) -> NDArray[np.intp]:
    """Extract HDBSCAN's infinite-row indices from reduced row sums."""
    values = np.asarray(reduced_row_sums, dtype=np.float64)
    return np.asarray(np.isinf(values).nonzero()[0], dtype=np.intp)

@register_atom(witness_hdbscan_finite_row_indices)
@icontract.require(lambda X: _sample_matrix_valid(X), "X must be a dense or sparse 2D sample matrix")
@icontract.ensure(lambda result, X: _strictly_increasing_indices(result, X.shape[0] if sp.issparse(X) else np.asarray(X).shape[0]), "finite row indices must be a sorted raw-index vector")
def hdbscan_finite_row_indices(
    X: object,
) -> NDArray[np.intp]:
    from sklearn.cluster._hdbscan.hdbscan import _get_finite_row_indices
    """Return HDBSCAN's purely finite row indices for dense or sparse input."""
    return np.asarray(_get_finite_row_indices(X), dtype=np.intp)

@register_atom(witness_hdbscan_internal_to_raw_map)
@icontract.require(lambda finite_index: _strictly_increasing_indices(finite_index, int(np.max(np.asarray(finite_index, dtype=np.intp)) + 1) if np.asarray(finite_index, dtype=np.intp).shape[0] else 1), "finite_index must be a sorted raw-index vector")
@icontract.ensure(lambda result, finite_index: _mapping_valid(result, finite_index), "mapping must assign each internal index to the corresponding raw finite index")
def hdbscan_internal_to_raw_map(
    finite_index: NDArray[np.intp],
) -> dict[int, int]:
    """Build HDBSCAN's internal-to-raw index map from finite row indices."""
    indices = np.asarray(finite_index, dtype=np.intp)
    return {int(i): int(raw) for i, raw in enumerate(indices)}

@register_atom(witness_hdbscan_nonfinite_raw_indices)
@icontract.require(lambda infinite_index: _index_vector_valid(infinite_index, int(np.max(np.asarray(infinite_index, dtype=np.intp)) + 1) if np.asarray(infinite_index, dtype=np.intp).shape[0] else 1), "infinite_index must be a raw-index vector")
@icontract.require(lambda missing_index: _index_vector_valid(missing_index, int(np.max(np.asarray(missing_index, dtype=np.intp)) + 1) if np.asarray(missing_index, dtype=np.intp).shape[0] else 1), "missing_index must be a raw-index vector")
@icontract.require(lambda infinite_index, missing_index: _disjoint_index_sets(infinite_index, missing_index), "infinite and missing raw-index vectors must be disjoint")
@icontract.ensure(lambda result: isinstance(result, set), "non-finite raw indices must be returned as a set")
def hdbscan_nonfinite_raw_indices(
    infinite_index: NDArray[np.intp],
    missing_index: NDArray[np.intp],
) -> set[int]:
    """Build HDBSCAN's non-finite raw-index set for linkage-tree remapping."""
    return set(int(x) for x in np.hstack([np.asarray(infinite_index, dtype=np.intp), np.asarray(missing_index, dtype=np.intp)]))

@register_atom(witness_hdbscan_remapped_labels)
@icontract.require(lambda raw_sample_count, finite_index, finite_labels, infinite_index, missing_index: _remap_inputs_valid(raw_sample_count, finite_index, finite_labels, infinite_index, missing_index), "raw sample count and index vectors must be compatible")
@icontract.require(lambda finite_index, finite_labels: np.asarray(finite_labels).ndim == 1 and np.asarray(finite_labels).shape[0] == np.asarray(finite_index, dtype=np.intp).shape[0] and np.issubdtype(np.asarray(finite_labels).dtype, np.integer), "finite_labels must align one-to-one with finite_index")
@icontract.ensure(lambda result, raw_sample_count: _label_vector_valid(result, raw_sample_count), "remapped labels must be an integer vector over the raw sample count")
def hdbscan_remapped_labels(
    raw_sample_count: int,
    finite_index: NDArray[np.intp],
    finite_labels: NDArray[np.int32],
    infinite_index: NDArray[np.intp],
    missing_index: NDArray[np.intp],
) -> NDArray[np.int32]:
    """Restore HDBSCAN labels onto the raw sample axis with non-finite encodings."""
    labels = np.empty(int(raw_sample_count), dtype=np.int32)
    labels[np.asarray(finite_index, dtype=np.intp)] = np.asarray(finite_labels, dtype=np.int32)
    labels[np.asarray(infinite_index, dtype=np.intp)] = _HDBSCAN_INFINITE_LABEL
    labels[np.asarray(missing_index, dtype=np.intp)] = _HDBSCAN_MISSING_LABEL
    return labels

@register_atom(witness_hdbscan_remapped_probabilities)
@icontract.require(lambda raw_sample_count, finite_index, finite_probabilities, infinite_index, missing_index: _remap_inputs_valid(raw_sample_count, finite_index, finite_probabilities, infinite_index, missing_index), "raw sample count and index vectors must be compatible")
@icontract.require(lambda finite_index, finite_probabilities: np.asarray(finite_probabilities, dtype=np.float64).ndim == 1 and np.asarray(finite_probabilities, dtype=np.float64).shape[0] == np.asarray(finite_index, dtype=np.intp).shape[0] and np.all(np.isfinite(np.asarray(finite_probabilities, dtype=np.float64))) and np.all((np.asarray(finite_probabilities, dtype=np.float64) >= 0.0) & (np.asarray(finite_probabilities, dtype=np.float64) <= 1.0)), "finite_probabilities must align one-to-one with finite_index and lie in [0, 1]")
@icontract.ensure(lambda result, raw_sample_count: _probability_vector_valid(result, raw_sample_count), "remapped probabilities must be a float vector over the raw sample count")
def hdbscan_remapped_probabilities(
    raw_sample_count: int,
    finite_index: NDArray[np.intp],
    finite_probabilities: NDArray[np.float64],
    infinite_index: NDArray[np.intp],
    missing_index: NDArray[np.intp],
) -> NDArray[np.float64]:
    """Restore HDBSCAN probabilities onto the raw sample axis with non-finite encodings."""
    probabilities = np.zeros(int(raw_sample_count), dtype=np.float64)
    probabilities[np.asarray(finite_index, dtype=np.intp)] = np.asarray(finite_probabilities, dtype=np.float64)
    probabilities[np.asarray(infinite_index, dtype=np.intp)] = _HDBSCAN_INFINITE_PROB
    probabilities[np.asarray(missing_index, dtype=np.intp)] = np.nan
    return probabilities
