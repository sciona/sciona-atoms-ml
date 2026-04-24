from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.feature_selection import VarianceThreshold


def test_selector_mixin_postfit_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.selector_mixin_postfit import (
        selector_feature_names_out,
        selector_inverse_transform_dense,
        selector_support_indices,
        selector_transform_dense,
    )

    assert callable(selector_support_indices)
    assert callable(selector_transform_dense)
    assert callable(selector_inverse_transform_dense)
    assert callable(selector_feature_names_out)


def test_selector_support_indices_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.selector_mixin_postfit import selector_support_indices

    X = np.array(
        [
            [0.0, 1.0, 5.0, 7.0],
            [0.0, 3.0, 5.0, 8.0],
            [0.0, 5.0, 6.0, 7.0],
        ],
        dtype=np.float64,
    )
    selector = VarianceThreshold(threshold=0.0).fit(X)
    support_mask = selector.get_support()

    assert np.array_equal(selector_support_indices(support_mask), selector.get_support(indices=True))


def test_selector_transform_dense_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.selector_mixin_postfit import selector_transform_dense

    X = np.array(
        [
            [0.0, 1.0, 5.0, 7.0],
            [0.0, 3.0, 5.0, 8.0],
            [0.0, 5.0, 6.0, 7.0],
        ],
        dtype=np.float64,
    )
    selector = VarianceThreshold(threshold=0.0).fit(X)
    support_mask = selector.get_support()

    assert np.array_equal(selector_transform_dense(X, support_mask), selector.transform(X))


def test_selector_transform_dense_matches_no_features_selected_shape() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.selector_mixin_postfit import selector_transform_dense

    class EmptySelector(SelectorMixin, BaseEstimator):
        def fit(self, X, y=None):  # type: ignore[override]
            self.n_features_in_ = X.shape[1]
            return self

        def _get_support_mask(self):
            return np.zeros(self.n_features_in_, dtype=np.bool_)

    X = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        ],
        dtype=np.float64,
    )
    support_mask = np.array([False, False, False], dtype=np.bool_)
    with pytest.warns(UserWarning):
        selector = EmptySelector().fit(X)
        expected = selector.transform(X)

    result = selector_transform_dense(X, support_mask)
    assert result.shape == (2, 0)
    assert np.array_equal(result, expected)


def test_selector_inverse_transform_dense_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.selector_mixin_postfit import (
        selector_inverse_transform_dense,
        selector_transform_dense,
    )

    X = np.array(
        [
            [0.0, 1.0, 5.0, 7.0],
            [0.0, 3.0, 5.0, 8.0],
            [0.0, 5.0, 6.0, 7.0],
        ],
        dtype=np.float64,
    )
    selector = VarianceThreshold(threshold=0.0).fit(X)
    support_mask = selector.get_support()
    X_selected = selector_transform_dense(X, support_mask)

    assert np.array_equal(selector_inverse_transform_dense(X_selected, support_mask), selector.inverse_transform(X_selected))


def test_selector_feature_names_out_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.selector_mixin_postfit import selector_feature_names_out

    X = np.array(
        [
            [0.0, 1.0, 5.0, 7.0],
            [0.0, 3.0, 5.0, 8.0],
            [0.0, 5.0, 6.0, 7.0],
        ],
        dtype=np.float64,
    )
    input_features = ("a", "b", "c", "d")
    selector = VarianceThreshold(threshold=0.0).fit(X)
    support_mask = selector.get_support()

    assert selector_feature_names_out(input_features, support_mask) == tuple(selector.get_feature_names_out(input_features))


def test_contracts_reject_invalid_selector_mixin_postfit_inputs() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.selector_mixin_postfit import (
        selector_feature_names_out,
        selector_inverse_transform_dense,
        selector_support_indices,
        selector_transform_dense,
    )

    with pytest.raises(ViolationError):
        selector_support_indices(np.array([1, 0, 1], dtype=np.int64))

    with pytest.raises(ViolationError):
        selector_transform_dense(
            np.ones((3, 4), dtype=np.float64),
            np.array([True, False, True], dtype=np.bool_),
        )

    with pytest.raises(ViolationError):
        selector_inverse_transform_dense(
            np.ones((2, 2), dtype=np.float64),
            np.array([True, False, False], dtype=np.bool_),
        )

    with pytest.raises(ViolationError):
        selector_feature_names_out(
            ("a", "b"),
            np.array([True, False, True], dtype=np.bool_),
        )
