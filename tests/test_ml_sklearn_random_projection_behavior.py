from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.exceptions import DataDimensionalityWarning
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection


def test_random_projection_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.random_projection import (
        gaussian_random_projection_fit,
        gaussian_random_projection_transform,
        random_projection_inverse_transform,
        sparse_random_projection_fit,
        sparse_random_projection_transform,
    )

    assert callable(gaussian_random_projection_fit)
    assert callable(gaussian_random_projection_transform)
    assert callable(random_projection_inverse_transform)
    assert callable(sparse_random_projection_fit)
    assert callable(sparse_random_projection_transform)


def test_gaussian_fit_transform_and_inverse_match_sklearn_dense() -> None:
    from sciona.atoms.ml.sklearn.random_projection import (
        gaussian_random_projection_fit,
        gaussian_random_projection_transform,
        random_projection_inverse_transform,
    )

    X = np.arange(30, dtype=np.float64).reshape(6, 5) / 10.0
    sklearn_projection = GaussianRandomProjection(
        n_components=3,
        compute_inverse_components=True,
        random_state=13,
    ).fit(X)
    state = gaussian_random_projection_fit(
        X,
        n_components=3,
        compute_inverse_components=True,
        random_state=13,
    )
    transformed = gaussian_random_projection_transform(X, state)

    assert state.n_components == sklearn_projection.n_components_
    assert state.n_features_in == sklearn_projection.n_features_in_
    assert np.allclose(state.components, sklearn_projection.components_)
    assert np.allclose(state.inverse_components, sklearn_projection.inverse_components_)
    assert np.allclose(transformed, sklearn_projection.transform(X))
    assert np.allclose(random_projection_inverse_transform(transformed, state), sklearn_projection.inverse_transform(transformed))


def test_gaussian_auto_components_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.random_projection import gaussian_random_projection_fit

    X = np.arange(200, dtype=np.float32).reshape(2, 100)
    sklearn_projection = GaussianRandomProjection(n_components="auto", eps=0.9, random_state=7).fit(X)
    state = gaussian_random_projection_fit(X, n_components="auto", eps=0.9, random_state=7)

    assert state.n_components == sklearn_projection.n_components_
    assert state.components.dtype == sklearn_projection.components_.dtype
    assert np.allclose(state.components, sklearn_projection.components_)


def test_gaussian_warns_when_components_exceed_features() -> None:
    from sciona.atoms.ml.sklearn.random_projection import gaussian_random_projection_fit

    X = np.ones((4, 3), dtype=np.float64)
    with pytest.warns(DataDimensionalityWarning):
        state = gaussian_random_projection_fit(X, n_components=5, random_state=0)
    with pytest.warns(DataDimensionalityWarning):
        sklearn_projection = GaussianRandomProjection(n_components=5, random_state=0).fit(X)

    assert state.components.shape == sklearn_projection.components_.shape


def test_sparse_fit_transform_match_sklearn_sparse_output() -> None:
    from sciona.atoms.ml.sklearn.random_projection import (
        sparse_random_projection_fit,
        sparse_random_projection_transform,
    )

    X = sp.csr_matrix(np.arange(48, dtype=np.float64).reshape(8, 6) / 10.0)
    sklearn_projection = SparseRandomProjection(
        n_components=4,
        density="auto",
        dense_output=False,
        random_state=5,
    ).fit(X)
    state = sparse_random_projection_fit(
        X,
        n_components=4,
        density="auto",
        dense_output=False,
        random_state=5,
    )
    transformed = sparse_random_projection_transform(X, state)
    sklearn_transformed = sklearn_projection.transform(X)

    assert state.n_components == sklearn_projection.n_components_
    assert state.n_features_in == sklearn_projection.n_features_in_
    assert state.density == sklearn_projection.density_
    assert np.allclose(state.components.toarray(), sklearn_projection.components_.toarray())
    assert sp.issparse(transformed)
    assert np.allclose(transformed.toarray(), sklearn_transformed.toarray())


def test_sparse_dense_output_and_inverse_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.random_projection import (
        random_projection_inverse_transform,
        sparse_random_projection_fit,
        sparse_random_projection_transform,
    )

    X = sp.csc_matrix(np.arange(35, dtype=np.float64).reshape(5, 7) / 7.0)
    sklearn_projection = SparseRandomProjection(
        n_components=3,
        density=1 / 3,
        dense_output=True,
        compute_inverse_components=True,
        random_state=17,
    ).fit(X)
    state = sparse_random_projection_fit(
        X,
        n_components=3,
        density=1 / 3,
        dense_output=True,
        compute_inverse_components=True,
        random_state=17,
    )
    transformed = sparse_random_projection_transform(X, state)

    assert np.allclose(state.components.toarray(), sklearn_projection.components_.toarray())
    assert isinstance(transformed, np.ndarray)
    assert np.allclose(transformed, sklearn_projection.transform(X))
    assert np.allclose(random_projection_inverse_transform(transformed, state), sklearn_projection.inverse_transform(transformed))


def test_sparse_density_one_dense_matrix_branch_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.random_projection import sparse_random_projection_fit

    X = np.arange(20, dtype=np.float32).reshape(4, 5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataDimensionalityWarning)
        sklearn_projection = SparseRandomProjection(n_components=6, density=1.0, random_state=21).fit(X)
        state = sparse_random_projection_fit(X, n_components=6, density=1.0, random_state=21)

    assert isinstance(state.components, np.ndarray)
    assert np.allclose(state.components, sklearn_projection.components_)
    assert state.components.dtype == sklearn_projection.components_.dtype


def test_auto_components_error_matches_sklearn_when_bound_exceeds_features() -> None:
    from sciona.atoms.ml.sklearn.random_projection import sparse_random_projection_fit

    X = np.ones((2, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="larger than the original space"):
        sparse_random_projection_fit(X, n_components="auto", eps=0.9, random_state=0)
    with pytest.raises(ValueError, match="larger than the original space"):
        SparseRandomProjection(n_components="auto", eps=0.9, random_state=0).fit(X)


def test_transform_rejects_wrong_feature_count() -> None:
    from sciona.atoms.ml.sklearn.random_projection import (
        gaussian_random_projection_fit,
        gaussian_random_projection_transform,
    )

    state = gaussian_random_projection_fit(np.ones((3, 4), dtype=np.float64), n_components=2, random_state=1)
    with pytest.raises(Exception):
        gaussian_random_projection_transform(np.ones((3, 5), dtype=np.float64), state)
