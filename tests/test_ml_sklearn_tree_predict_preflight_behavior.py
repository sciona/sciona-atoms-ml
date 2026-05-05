from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_tree_predict_preflight_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.tree.predict_preflight import (
        tree_predict_ensure_all_finite_mode,
        tree_predict_require_sparse_int32_indices,
        tree_predict_use_check_input_branch,
    )

    assert callable(tree_predict_use_check_input_branch)
    assert callable(tree_predict_ensure_all_finite_mode)
    assert callable(tree_predict_require_sparse_int32_indices)


def test_tree_predict_preflight_matches_sklearn_rules() -> None:
    from sciona.atoms.ml.sklearn.tree.predict_preflight import (
        tree_predict_ensure_all_finite_mode,
        tree_predict_require_sparse_int32_indices,
        tree_predict_use_check_input_branch,
    )

    assert tree_predict_use_check_input_branch(True) is True
    assert tree_predict_use_check_input_branch(False) is False
    assert tree_predict_ensure_all_finite_mode(False) is True
    assert tree_predict_ensure_all_finite_mode(True) == "allow-nan"

    indices = np.array([0, 1, 0, 1], dtype=np.intc)
    indptr = np.array([0, 2, 4], dtype=np.intc)
    assert tree_predict_require_sparse_int32_indices(indices, indptr) is True

    with pytest.raises(ValueError, match="No support for np.int64 index based sparse matrices"):
        tree_predict_require_sparse_int32_indices(
            indices.astype(np.int64),
            indptr.astype(np.int64),
        )


def test_contracts_reject_invalid_tree_predict_preflight_inputs() -> None:
    from sciona.atoms.ml.sklearn.tree.predict_preflight import (
        tree_predict_require_sparse_int32_indices,
        tree_predict_use_check_input_branch,
    )

    with pytest.raises(ViolationError):
        tree_predict_use_check_input_branch(1)

    with pytest.raises(ViolationError):
        tree_predict_require_sparse_int32_indices(
            np.array([[0, 1]], dtype=np.intc),
            np.array([0, 2], dtype=np.intc),
        )

