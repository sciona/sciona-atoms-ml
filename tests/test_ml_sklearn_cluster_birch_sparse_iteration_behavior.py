from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from icontract import ViolationError
from sklearn.cluster._birch import _iterate_sparse_X


def _csr() -> sp.csr_matrix:
    return sp.csr_matrix(
        np.array(
            [
                [0.0, 1.0, 0.0, 2.0],
                [3.0, 0.0, 0.0, 0.0],
                [0.0, 4.0, 5.0, 0.0],
            ],
            dtype=np.float64,
        )
    )


def test_birch_sparse_iteration_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.birch_sparse_iteration import (
        birch_sparse_dense_row,
        birch_sparse_dense_rows,
        birch_sparse_row_bounds,
    )

    assert callable(birch_sparse_row_bounds)
    assert callable(birch_sparse_dense_row)
    assert callable(birch_sparse_dense_rows)


def test_birch_sparse_iteration_matches_private_helper() -> None:
    from sciona.atoms.ml.sklearn.cluster.birch_sparse_iteration import (
        birch_sparse_dense_row,
        birch_sparse_dense_rows,
        birch_sparse_row_bounds,
    )

    X = _csr()
    expected_rows = np.asarray(list(_iterate_sparse_X(X)), dtype=np.float64)

    startptr, endptr = birch_sparse_row_bounds(X.indptr, 0)
    first_row = birch_sparse_dense_row(
        X.shape[1],
        X.indices[startptr:endptr],
        X.data[startptr:endptr],
    )
    actual_rows = birch_sparse_dense_rows(X.indptr, X.indices, X.data, X.shape[1])

    assert startptr == int(X.indptr[0])
    assert endptr == int(X.indptr[1])
    assert np.array_equal(first_row, expected_rows[0])
    assert np.array_equal(actual_rows, expected_rows)


def test_birch_sparse_iteration_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.cluster.birch_sparse_iteration import (
        birch_sparse_dense_row,
        birch_sparse_dense_rows,
        birch_sparse_row_bounds,
    )

    with pytest.raises(ViolationError):
        birch_sparse_row_bounds(np.array([0], dtype=np.int64), 0)

    with pytest.raises(ViolationError):
        birch_sparse_dense_row(0, np.array([], dtype=np.int64), np.array([], dtype=np.float64))

    with pytest.raises(ViolationError):
        birch_sparse_dense_rows(
            np.array([0, 2], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([1.0, 2.0], dtype=np.float64),
            3,
        )
