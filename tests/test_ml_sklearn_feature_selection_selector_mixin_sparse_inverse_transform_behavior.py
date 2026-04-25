from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from icontract import ViolationError
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.feature_selection import VarianceThreshold


def test_selector_mixin_sparse_inverse_transform_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.selector_mixin_sparse_inverse_transform import (
        selector_inverse_transform_csc,
    )

    assert callable(selector_inverse_transform_csc)


def test_selector_inverse_transform_csc_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.selector_mixin_sparse_inverse_transform import (
        selector_inverse_transform_csc,
    )

    X_dense = np.array(
        [
            [0.0, 1.0, 5.0, 0.0],
            [0.0, 3.0, 5.0, 8.0],
            [0.0, 5.0, 6.0, 0.0],
        ],
        dtype=np.float64,
    )
    X = sp.csr_matrix(X_dense)
    selector = VarianceThreshold(threshold=0.0).fit(X)
    support_mask = selector.get_support()
    X_selected = selector.transform(X)

    result = selector_inverse_transform_csc(X_selected, support_mask)
    expected = selector.inverse_transform(X_selected)

    assert sp.isspmatrix_csc(result)
    assert np.array_equal(result.toarray(), expected.toarray())
    assert np.array_equal(result.indptr, expected.indptr)
    assert np.array_equal(result.indices, expected.indices)
    assert np.array_equal(result.data, expected.data)


def test_contracts_reject_invalid_sparse_selector_inputs() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.selector_mixin_sparse_inverse_transform import (
        selector_inverse_transform_csc,
    )

    with pytest.raises(ViolationError):
        selector_inverse_transform_csc(
            sp.csr_matrix(np.ones((2, 2), dtype=np.float64)),
            np.array([True, False, False], dtype=np.bool_),
        )

    with pytest.raises(ViolationError):
        selector_inverse_transform_csc(
            sp.csr_matrix([[np.inf, 1.0]], dtype=np.float64),
            np.array([True, True], dtype=np.bool_),
        )
