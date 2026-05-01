from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp


def test_hdbscan_nonfinite_remapping_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.hdbscan_nonfinite_remapping import (
        hdbscan_finite_row_indices,
        hdbscan_infinite_indices,
        hdbscan_internal_to_raw_map,
        hdbscan_missing_indices,
        hdbscan_nonfinite_raw_indices,
        hdbscan_remapped_labels,
        hdbscan_remapped_probabilities,
    )

    assert callable(hdbscan_missing_indices)
    assert callable(hdbscan_infinite_indices)
    assert callable(hdbscan_finite_row_indices)
    assert callable(hdbscan_internal_to_raw_map)
    assert callable(hdbscan_nonfinite_raw_indices)
    assert callable(hdbscan_remapped_labels)
    assert callable(hdbscan_remapped_probabilities)


def test_hdbscan_nonfinite_index_helpers_match_source_logic() -> None:
    from sciona.atoms.ml.sklearn.cluster.hdbscan_nonfinite_remapping import (
        hdbscan_finite_row_indices,
        hdbscan_infinite_indices,
        hdbscan_internal_to_raw_map,
        hdbscan_missing_indices,
        hdbscan_nonfinite_raw_indices,
    )

    reduced = np.array([1.0, np.nan, np.inf, 2.0], dtype=np.float64)
    X_dense = np.array(
        [
            [1.0, 0.0],
            [np.nan, 1.0],
            [np.inf, 1.0],
            [2.0, 3.0],
        ],
        dtype=np.float64,
    )
    X_sparse = sp.csr_matrix(
        np.array(
            [
                [1.0, 0.0],
                [2.0, 3.0],
                [4.0, np.inf],
                [5.0, 6.0],
            ],
            dtype=np.float64,
        )
    )

    missing = hdbscan_missing_indices(reduced)
    infinite = hdbscan_infinite_indices(reduced)
    assert np.array_equal(missing, np.array([1], dtype=np.intp))
    assert np.array_equal(infinite, np.array([2], dtype=np.intp))
    assert hdbscan_nonfinite_raw_indices(infinite, missing) == {1, 2}

    finite_dense = hdbscan_finite_row_indices(X_dense)
    finite_sparse = hdbscan_finite_row_indices(X_sparse)
    assert np.array_equal(finite_dense, np.array([0, 3], dtype=np.intp))
    assert np.array_equal(finite_sparse, np.array([0, 1, 3], dtype=np.intp))
    assert hdbscan_internal_to_raw_map(finite_dense) == {0: 0, 1: 3}


def test_hdbscan_nonfinite_output_remapping_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.cluster.hdbscan_nonfinite_remapping import (
        hdbscan_remapped_labels,
        hdbscan_remapped_probabilities,
    )

    finite_index = np.array([0, 3], dtype=np.intp)
    finite_labels = np.array([5, 7], dtype=np.int32)
    finite_probabilities = np.array([0.8, 0.25], dtype=np.float64)
    infinite_index = np.array([2], dtype=np.intp)
    missing_index = np.array([1], dtype=np.intp)

    labels = hdbscan_remapped_labels(4, finite_index, finite_labels, infinite_index, missing_index)
    probabilities = hdbscan_remapped_probabilities(
        4,
        finite_index,
        finite_probabilities,
        infinite_index,
        missing_index,
    )

    assert np.array_equal(labels, np.array([5, -3, -2, 7], dtype=np.int32))
    assert probabilities[0] == pytest.approx(0.8)
    assert np.isnan(probabilities[1])
    assert probabilities[2] == pytest.approx(0.0)
    assert probabilities[3] == pytest.approx(0.25)


def test_hdbscan_nonfinite_remapping_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.cluster.hdbscan_nonfinite_remapping import (
        hdbscan_internal_to_raw_map,
        hdbscan_missing_indices,
        hdbscan_remapped_labels,
        hdbscan_remapped_probabilities,
    )

    with pytest.raises(Exception):
        hdbscan_missing_indices(np.array([[1.0, np.nan]], dtype=np.float64))

    with pytest.raises(Exception):
        hdbscan_internal_to_raw_map(np.array([3, 2], dtype=np.intp))

    with pytest.raises(Exception):
        hdbscan_remapped_labels(
            4,
            np.array([0, 3], dtype=np.intp),
            np.array([1], dtype=np.int32),
            np.array([2], dtype=np.intp),
            np.array([1], dtype=np.intp),
        )

    with pytest.raises(Exception):
        hdbscan_remapped_probabilities(
            4,
            np.array([0, 3], dtype=np.intp),
            np.array([1.2, 0.4], dtype=np.float64),
            np.array([2], dtype=np.intp),
            np.array([1], dtype=np.intp),
        )
