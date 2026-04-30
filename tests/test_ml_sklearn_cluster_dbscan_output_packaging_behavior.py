from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from icontract import ViolationError
from sklearn.cluster import DBSCAN as SklearnDBSCAN


def _data() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [0.0, 0.1],
            [0.2, 0.0],
            [5.0, 5.0],
            [5.1, 5.0],
            [10.0, 10.0],
        ],
        dtype=np.float64,
    )


def test_dbscan_output_packaging_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.dbscan_output_packaging import (
        dbscan_core_sample_indices,
        dbscan_sparse_core_components,
    )

    assert callable(dbscan_core_sample_indices)
    assert callable(dbscan_sparse_core_components)


def test_dbscan_core_sample_indices_matches_sklearn_fit_shell() -> None:
    from sciona.atoms.ml.sklearn.cluster.dbscan_output_packaging import dbscan_core_sample_indices
    from sciona.atoms.ml.sklearn.cluster.dbscan_fit_bookkeeping import dbscan_core_sample_mask

    X = _data()
    expected = SklearnDBSCAN(eps=0.35, min_samples=2, metric="euclidean").fit(X)
    neighbor_mass = np.array([3, 3, 3, 2, 2, 1], dtype=np.int64)
    core_mask = dbscan_core_sample_mask(neighbor_mass, 2)

    assert np.array_equal(dbscan_core_sample_indices(core_mask), expected.core_sample_indices_.astype(np.intp))


def test_dbscan_sparse_core_components_matches_sklearn_sparse_fit() -> None:
    from sciona.atoms.ml.sklearn.cluster.dbscan_output_packaging import (
        dbscan_core_sample_indices,
        dbscan_sparse_core_components,
    )
    from sciona.atoms.ml.sklearn.cluster.dbscan_fit_bookkeeping import dbscan_core_sample_mask

    X_sparse = sp.csr_matrix(_data())
    expected = SklearnDBSCAN(eps=0.35, min_samples=2, metric="euclidean").fit(X_sparse)
    neighbor_mass = np.array([3, 3, 3, 2, 2, 1], dtype=np.int64)
    core_indices = dbscan_core_sample_indices(dbscan_core_sample_mask(neighbor_mass, 2))

    actual = dbscan_sparse_core_components(X_sparse, core_indices)

    assert sp.issparse(actual)
    assert actual.format == expected.components_.format
    assert np.array_equal(actual.toarray(), expected.components_.toarray())


def test_dbscan_output_packaging_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.cluster.dbscan_output_packaging import (
        dbscan_core_sample_indices,
        dbscan_sparse_core_components,
    )

    with pytest.raises(ViolationError):
        dbscan_core_sample_indices(np.array([], dtype=np.uint8))

    with pytest.raises(ViolationError):
        dbscan_sparse_core_components(np.asarray(_data()), np.array([0, 1], dtype=np.intp))

    with pytest.raises(ViolationError):
        dbscan_sparse_core_components(sp.csr_matrix(_data()), np.array([99], dtype=np.intp))
