from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from icontract import ViolationError
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.feature_selection import VarianceThreshold


def test_selector_mixin_sparse_transform_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.selector_mixin_sparse_transform import (
        selector_transform_sparse,
    )

    assert callable(selector_transform_sparse)


def test_selector_transform_sparse_matches_sklearn_selected_columns() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.selector_mixin_sparse_transform import (
        selector_transform_sparse,
    )

    X = sp.csr_matrix(
        [
            [0.0, 1.0, 5.0, 0.0],
            [0.0, 3.0, 5.0, 8.0],
            [0.0, 5.0, 6.0, 0.0],
        ],
        dtype=np.float64,
    )
    selector = VarianceThreshold(threshold=0.0).fit(X)
    support_mask = selector.get_support()

    result = selector_transform_sparse(X, support_mask)
    expected = selector._transform(X)

    assert sp.isspmatrix_csr(result)
    assert np.array_equal(result.toarray(), expected.toarray())


def test_selector_transform_sparse_matches_sklearn_empty_dense_fallback() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.selector_mixin_sparse_transform import (
        selector_transform_sparse,
    )

    class EmptySelector(SelectorMixin, BaseEstimator):
        def fit(self, X, y=None):  # type: ignore[override]
            self.n_features_in_ = X.shape[1]
            return self

        def _get_support_mask(self):
            return np.zeros(self.n_features_in_, dtype=np.bool_)

    X = sp.csr_matrix(np.ones((2, 3), dtype=np.float64))
    selector = EmptySelector().fit(X)
    support_mask = selector.get_support()
    with pytest.warns(UserWarning):
        expected = selector._transform(X)
    result = selector_transform_sparse(X, support_mask)

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 0)
    assert np.array_equal(result, expected)


def test_contracts_reject_invalid_sparse_selector_transform_inputs() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.selector_mixin_sparse_transform import (
        selector_transform_sparse,
    )

    with pytest.raises(ViolationError):
        selector_transform_sparse(
            sp.csr_matrix(np.ones((2, 2), dtype=np.float64)),
            np.array([True, False, False], dtype=np.bool_),
        )

    with pytest.raises(ViolationError):
        selector_transform_sparse(
            sp.csr_matrix([[np.inf, 1.0]], dtype=np.float64),
            np.array([True, True], dtype=np.bool_),
        )
