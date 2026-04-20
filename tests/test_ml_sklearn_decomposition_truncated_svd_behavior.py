from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import TruncatedSVD as SklearnTruncatedSVD


def _data() -> np.ndarray:
    return np.array(
        [
            [0.0, 1.0, 3.0, 2.0, 0.5],
            [1.0, 2.0, 4.0, 3.0, 1.5],
            [2.0, 4.0, 7.0, 5.0, 2.0],
            [4.0, 8.0, 9.0, 7.0, 3.5],
            [5.0, 9.0, 12.0, 11.0, 4.0],
            [6.0, 11.0, 13.0, 13.0, 6.5],
        ],
        dtype=np.float64,
    )


def test_truncated_svd_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.decomposition import (
        TruncatedSVDState,
        truncated_svd_fit,
        truncated_svd_inverse_transform,
        truncated_svd_transform,
    )

    assert TruncatedSVDState is not None
    assert callable(truncated_svd_fit)
    assert callable(truncated_svd_transform)
    assert callable(truncated_svd_inverse_transform)


def test_truncated_svd_fit_and_transform_match_sklearn_randomized() -> None:
    from sciona.atoms.ml.sklearn.decomposition import (
        truncated_svd_fit,
        truncated_svd_inverse_transform,
        truncated_svd_transform,
    )

    X = _data()
    state = truncated_svd_fit(X, n_components=3, n_iter=7, random_state=42)
    expected = SklearnTruncatedSVD(n_components=3, n_iter=7, random_state=42).fit(X)

    assert np.allclose(state.components, expected.components_)
    assert np.allclose(state.explained_variance, expected.explained_variance_)
    assert np.allclose(state.explained_variance_ratio, expected.explained_variance_ratio_)
    assert np.allclose(state.singular_values, expected.singular_values_)
    assert state.n_components == expected.components_.shape[0]
    assert state.n_features_in == expected.n_features_in_
    assert state.algorithm == "randomized"

    query = X[[0, 2, 5]]
    transformed = truncated_svd_transform(query, state)
    assert np.allclose(transformed, expected.transform(query))
    assert np.allclose(truncated_svd_inverse_transform(transformed, state), expected.inverse_transform(transformed))


def test_truncated_svd_fit_with_power_iteration_options_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.decomposition import truncated_svd_fit, truncated_svd_transform

    X = _data()
    state = truncated_svd_fit(
        X,
        n_components=2,
        n_iter=3,
        n_oversamples=4,
        power_iteration_normalizer="OR",
        random_state=0,
    )
    expected = SklearnTruncatedSVD(
        n_components=2,
        n_iter=3,
        n_oversamples=4,
        power_iteration_normalizer="OR",
        random_state=0,
    ).fit(X)

    assert np.allclose(state.components, expected.components_)
    assert np.allclose(truncated_svd_transform(X[:4], state), expected.transform(X[:4]))


def test_truncated_svd_rejects_unsupported_inputs() -> None:
    from sciona.atoms.ml.sklearn.decomposition import truncated_svd_fit

    X = _data()
    with pytest.raises(Exception):
        truncated_svd_fit(X, algorithm="arpack")

    with pytest.raises(Exception):
        truncated_svd_fit(X, n_components=0)

    with pytest.raises(Exception):
        truncated_svd_fit(X, n_components=min(X.shape) + 1)

    with pytest.raises(Exception):
        truncated_svd_fit(np.ones((4, 3), dtype=np.float64))
