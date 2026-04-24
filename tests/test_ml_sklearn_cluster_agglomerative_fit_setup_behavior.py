from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from icontract import ViolationError
from sklearn.cluster import _agglomerative as sklearn_agglomerative
from sklearn.neighbors import kneighbors_graph
from sklearn.utils.validation import check_array


def test_agglomerative_fit_setup_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.agglomerative_fit_setup import (
        agglomerative_fit_prepare_connectivity,
        agglomerative_fit_select_tree_builder,
    )

    assert callable(agglomerative_fit_select_tree_builder)
    assert callable(agglomerative_fit_prepare_connectivity)


def test_agglomerative_fit_select_tree_builder_matches_sklearn_mapping() -> None:
    from sciona.atoms.ml.sklearn.cluster.agglomerative_fit_setup import (
        agglomerative_fit_select_tree_builder,
    )

    for linkage in ("ward", "complete", "average", "single"):
        assert agglomerative_fit_select_tree_builder(linkage) is sklearn_agglomerative._TREE_BUILDERS[linkage]


def test_agglomerative_fit_prepare_connectivity_matches_check_array_for_matrix() -> None:
    from sciona.atoms.ml.sklearn.cluster.agglomerative_fit_setup import (
        agglomerative_fit_prepare_connectivity,
    )

    X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]], dtype=np.float64)
    raw = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    result = agglomerative_fit_prepare_connectivity(X, raw)
    expected = check_array(raw, accept_sparse=["csr", "coo", "lil"])
    assert np.array_equal(result, expected)


def test_agglomerative_fit_prepare_connectivity_executes_callable_and_preserves_sparse_type() -> None:
    from sciona.atoms.ml.sklearn.cluster.agglomerative_fit_setup import (
        agglomerative_fit_prepare_connectivity,
    )

    X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [1.5, 0.5]], dtype=np.float64)

    def connectivity_fn(values: np.ndarray) -> sp.csr_matrix:
        return kneighbors_graph(values, n_neighbors=2, include_self=False).tocsr()

    result = agglomerative_fit_prepare_connectivity(X, connectivity_fn)
    expected = check_array(connectivity_fn(X), accept_sparse=["csr", "coo", "lil"])
    assert sp.issparse(result)
    assert result.format == expected.format
    assert np.array_equal(result.toarray(), expected.toarray())


def test_agglomerative_fit_prepare_connectivity_supports_none() -> None:
    from sciona.atoms.ml.sklearn.cluster.agglomerative_fit_setup import (
        agglomerative_fit_prepare_connectivity,
    )

    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    assert agglomerative_fit_prepare_connectivity(X, None) is None


def test_contracts_reject_invalid_agglomerative_fit_setup_inputs() -> None:
    from sciona.atoms.ml.sklearn.cluster.agglomerative_fit_setup import (
        agglomerative_fit_prepare_connectivity,
        agglomerative_fit_select_tree_builder,
    )

    with pytest.raises(ViolationError):
        agglomerative_fit_select_tree_builder("median")

    with pytest.raises(ViolationError):
        agglomerative_fit_prepare_connectivity(
            np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
            np.array([[1.0, 0.0, 1.0]], dtype=np.float64),
        )
