from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
import pytest
from sklearn.feature_selection import VarianceThreshold


def test_variance_threshold_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.variance_threshold import (
        variance_threshold_fit,
        variance_threshold_support_mask,
        variance_threshold_transform,
    )

    assert callable(variance_threshold_fit)
    assert callable(variance_threshold_support_mask)
    assert callable(variance_threshold_transform)


def test_fit_support_and_transform_match_sklearn_dense() -> None:
    from sciona.atoms.ml.sklearn.variance_threshold import (
        variance_threshold_fit,
        variance_threshold_support_mask,
        variance_threshold_transform,
    )

    X = np.array([[0.0, 2.0, 0.0, 3.0], [0.0, 1.0, 4.0, 3.0], [0.0, 1.0, 1.0, 3.0]])
    sklearn_selector = VarianceThreshold().fit(X)
    state = variance_threshold_fit(X)

    assert np.allclose(state.variances, sklearn_selector.variances_)
    assert np.array_equal(variance_threshold_support_mask(state), sklearn_selector.get_support())
    assert np.array_equal(variance_threshold_transform(X, state), sklearn_selector.transform(X))


def test_fit_support_and_transform_match_sklearn_thresholded_sparse() -> None:
    from sciona.atoms.ml.sklearn.variance_threshold import (
        variance_threshold_fit,
        variance_threshold_support_mask,
        variance_threshold_transform,
    )

    X = sp.csr_matrix(
        np.array([[0.0, 2.0, 0.0, 3.0], [0.0, 1.0, 4.0, 3.0], [1.0, 1.0, 1.0, 3.0]])
    )
    threshold = 0.15
    sklearn_selector = VarianceThreshold(threshold=threshold).fit(X)
    state = variance_threshold_fit(X, threshold=threshold)

    assert np.allclose(state.variances, sklearn_selector.variances_)
    assert np.array_equal(variance_threshold_support_mask(state), sklearn_selector.get_support())
    assert np.array_equal(
        variance_threshold_transform(X, state).toarray(),
        sklearn_selector.transform(X).toarray(),
    )


def test_nan_variance_behavior_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.variance_threshold import variance_threshold_fit

    X = np.array([[np.nan, 1.0, 0.0], [np.nan, 2.0, 0.0], [np.nan, 3.0, 1.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sklearn_selector = VarianceThreshold().fit(X)
        state = variance_threshold_fit(X)

    assert np.allclose(state.variances, sklearn_selector.variances_, equal_nan=True)


def test_no_selected_feature_error_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.variance_threshold import variance_threshold_fit

    X = np.ones((3, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="No feature in X meets the variance threshold"):
        variance_threshold_fit(X)
    with pytest.raises(ValueError, match="No feature in X meets the variance threshold"):
        VarianceThreshold().fit(X)


def test_transform_rejects_wrong_feature_count() -> None:
    from sciona.atoms.ml.sklearn.variance_threshold import (
        variance_threshold_fit,
        variance_threshold_transform,
    )

    state = variance_threshold_fit(np.array([[0.0, 1.0], [1.0, 3.0]]))
    with pytest.raises(Exception):
        variance_threshold_transform(np.array([[1.0, 2.0, 3.0]]), state)
