from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier


def test_forest_path_outputs_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_path_outputs import (
        forest_apply_leaf_matrix,
        forest_decision_path_csr,
        forest_decision_path_node_ptr,
    )

    assert callable(forest_apply_leaf_matrix)
    assert callable(forest_decision_path_csr)
    assert callable(forest_decision_path_node_ptr)


def test_forest_apply_leaf_matrix_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_path_outputs import forest_apply_leaf_matrix

    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 0, 1, 1], dtype=np.int64)

    forest = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=0).fit(X, y)
    leaf_vectors = tuple(tree.apply(X, check_input=False).astype(np.int64) for tree in forest.estimators_)

    result = forest_apply_leaf_matrix(leaf_vectors)

    assert np.array_equal(result, forest.apply(X))


def test_forest_decision_path_outputs_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_path_outputs import (
        forest_decision_path_csr,
        forest_decision_path_node_ptr,
    )

    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 0, 1, 1], dtype=np.int64)

    forest = RandomForestClassifier(n_estimators=4, max_depth=2, random_state=1).fit(X, y)
    indicators = tuple(tree.decision_path(X, check_input=False) for tree in forest.estimators_)

    actual_indicator = forest_decision_path_csr(indicators)
    actual_ptr = forest_decision_path_node_ptr(indicators)
    expected_indicator, expected_ptr = forest.decision_path(X)

    assert sp.isspmatrix_csr(actual_indicator)
    assert np.array_equal(actual_indicator.toarray(), expected_indicator.toarray())
    assert np.array_equal(actual_ptr, expected_ptr)


def test_contracts_reject_invalid_forest_path_outputs_inputs() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_path_outputs import (
        forest_apply_leaf_matrix,
        forest_decision_path_csr,
        forest_decision_path_node_ptr,
    )

    try:
        forest_apply_leaf_matrix((np.array([1, 2], dtype=np.int64), np.array([1], dtype=np.int64)))
    except Exception as exc:
        assert exc.__class__.__name__ in {"ViolationError", "PreconditionError"}
    else:
        raise AssertionError("expected contract failure for misaligned leaf vectors")

    bad_dense = np.array([[1.0, 0.0]], dtype=np.float64)
    for fn in (forest_decision_path_node_ptr, forest_decision_path_csr):
        try:
            fn((bad_dense,))  # type: ignore[arg-type]
        except Exception as exc:
            assert exc.__class__.__name__ in {"ViolationError", "PreconditionError"}
        else:
            raise AssertionError("expected contract failure for non-sparse indicator input")
