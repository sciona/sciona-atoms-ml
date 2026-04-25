from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_forest_predict_preflight_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_predict_preflight import (
        forest_predict_ensure_all_finite_mode,
        forest_predict_require_sparse_int32_indices,
    )

    assert callable(forest_predict_ensure_all_finite_mode)
    assert callable(forest_predict_require_sparse_int32_indices)


def test_forest_predict_ensure_all_finite_mode_matches_sklearn_rule() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_predict_preflight import (
        forest_predict_ensure_all_finite_mode,
    )

    assert forest_predict_ensure_all_finite_mode(False) is True
    assert forest_predict_ensure_all_finite_mode(True) == "allow-nan"


def test_forest_predict_require_sparse_int32_indices_matches_sklearn_guard() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_predict_preflight import (
        forest_predict_require_sparse_int32_indices,
    )

    indices = np.array([0, 1, 0, 1], dtype=np.intc)
    indptr = np.array([0, 2, 4], dtype=np.intc)
    assert forest_predict_require_sparse_int32_indices(indices, indptr) is True

    with pytest.raises(ValueError, match="No support for np.int64 index based sparse matrices"):
        forest_predict_require_sparse_int32_indices(
            indices.astype(np.int64),
            indptr.astype(np.int64),
        )


def test_contracts_reject_invalid_forest_predict_preflight_inputs() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_predict_preflight import (
        forest_predict_require_sparse_int32_indices,
    )

    with pytest.raises(ViolationError):
        forest_predict_require_sparse_int32_indices(
            np.array([[0, 1]], dtype=np.intc),
            np.array([0, 2], dtype=np.intc),
        )

    with pytest.raises(ViolationError):
        forest_predict_require_sparse_int32_indices(
            np.array([0, 1], dtype=np.float64),
            np.array([0, 2], dtype=np.intc),
        )
