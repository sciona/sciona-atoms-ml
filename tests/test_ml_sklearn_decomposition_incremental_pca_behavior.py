from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import IncrementalPCA as SklearnIncrementalPCA


def _data() -> np.ndarray:
    return np.array(
        [
            [0.0, 1.0, 3.0, 2.0],
            [1.0, 2.0, 4.0, 3.0],
            [2.0, 4.0, 7.0, 5.0],
            [4.0, 8.0, 9.0, 7.0],
            [5.0, 9.0, 12.0, 11.0],
            [6.0, 11.0, 13.0, 13.0],
        ],
        dtype=np.float64,
    )


def _assert_state_matches_sklearn(state, expected: SklearnIncrementalPCA) -> None:
    assert np.allclose(state.components, expected.components_)
    assert np.allclose(state.explained_variance, expected.explained_variance_)
    assert np.allclose(state.explained_variance_ratio, expected.explained_variance_ratio_)
    assert np.allclose(state.singular_values, expected.singular_values_)
    assert np.allclose(state.mean, expected.mean_)
    assert np.allclose(state.var, expected.var_)
    assert np.isclose(state.noise_variance, expected.noise_variance_)
    assert state.n_samples_seen == expected.n_samples_seen_
    assert state.n_components == expected.n_components_
    assert state.n_features_in == expected.n_features_in_


def test_incremental_pca_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.decomposition import (
        IncrementalPCAState,
        incremental_pca_inverse_transform,
        incremental_pca_partial_fit,
        incremental_pca_transform,
    )

    assert IncrementalPCAState is not None
    assert callable(incremental_pca_partial_fit)
    assert callable(incremental_pca_transform)
    assert callable(incremental_pca_inverse_transform)


def test_incremental_pca_first_batch_matches_sklearn_partial_fit() -> None:
    from sciona.atoms.ml.sklearn.decomposition import incremental_pca_partial_fit

    X = _data()
    state = incremental_pca_partial_fit(X[:3], n_components=2)
    expected = SklearnIncrementalPCA(n_components=2).partial_fit(X[:3])

    _assert_state_matches_sklearn(state, expected)
    assert state.whiten is False


def test_incremental_pca_second_batch_transform_and_inverse_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.decomposition import (
        incremental_pca_inverse_transform,
        incremental_pca_partial_fit,
        incremental_pca_transform,
    )

    X = _data()
    state = incremental_pca_partial_fit(X[:3], n_components=2)
    state = incremental_pca_partial_fit(X[3:], n_components=2, state=state)
    expected = SklearnIncrementalPCA(n_components=2).partial_fit(X[:3]).partial_fit(X[3:])

    _assert_state_matches_sklearn(state, expected)
    query = X[[0, 2, 5]]
    transformed = incremental_pca_transform(query, state)
    assert np.allclose(transformed, expected.transform(query))
    assert np.allclose(incremental_pca_inverse_transform(transformed, state), expected.inverse_transform(transformed))


def test_incremental_pca_whitened_transform_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.decomposition import (
        incremental_pca_inverse_transform,
        incremental_pca_partial_fit,
        incremental_pca_transform,
    )

    X = _data()
    state = incremental_pca_partial_fit(X[:3], n_components=2, whiten=True)
    state = incremental_pca_partial_fit(X[3:], state=state)
    expected = SklearnIncrementalPCA(n_components=2, whiten=True).partial_fit(X[:3]).partial_fit(X[3:])

    query = X[:4]
    transformed = incremental_pca_transform(query, state)
    assert np.allclose(transformed, expected.transform(query))
    assert np.allclose(incremental_pca_inverse_transform(transformed, state), expected.inverse_transform(transformed))


def test_incremental_pca_rejects_unsupported_inputs() -> None:
    from sciona.atoms.ml.sklearn.decomposition import incremental_pca_partial_fit

    X = _data()
    with pytest.raises(Exception):
        incremental_pca_partial_fit(X[:3], n_components=0)

    with pytest.raises(Exception):
        incremental_pca_partial_fit(X[:3], n_components=4)

    state = incremental_pca_partial_fit(X[:3], n_components=2)
    with pytest.raises(Exception):
        incremental_pca_partial_fit(X[3:], n_components=3, state=state)

    with pytest.raises(Exception):
        incremental_pca_partial_fit(np.ones((4, 3), dtype=np.float64))
