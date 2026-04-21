from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import FastICA as SklearnFastICA


def _data() -> np.ndarray:
    return np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 1.5],
            [2.0, 1.0, 0.5],
            [3.0, 4.0, 3.5],
            [4.0, 3.0, 2.5],
            [5.0, 5.0, 4.0],
        ],
        dtype=np.float64,
    )


def test_fastica_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.decomposition import (
        FastICAState,
        fastica_fit,
        fastica_inverse_transform,
        fastica_transform,
    )

    assert FastICAState is not None
    assert callable(fastica_fit)
    assert callable(fastica_transform)
    assert callable(fastica_inverse_transform)


def test_fastica_parallel_unit_variance_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.decomposition import fastica_fit, fastica_inverse_transform, fastica_transform

    X = _data()
    state = fastica_fit(X, n_components=2, random_state=0, max_iter=500, tol=1e-5)
    expected = SklearnFastICA(n_components=2, random_state=0, max_iter=500, tol=1e-5, whiten="unit-variance").fit(X)

    assert np.allclose(state.components, expected.components_)
    assert np.allclose(state.mixing, expected.mixing_)
    assert np.allclose(state.unmixing, expected._unmixing)
    assert np.allclose(state.whitening, expected.whitening_)
    assert np.allclose(state.mean, expected.mean_)
    assert state.n_iter == expected.n_iter_
    assert state.n_features_in == expected.n_features_in_
    assert np.allclose(fastica_transform(X, state), expected.transform(X))
    assert np.allclose(
        fastica_inverse_transform(fastica_transform(X, state), state),
        expected.inverse_transform(expected.transform(X)),
    )


def test_fastica_deflation_arbitrary_variance_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.decomposition import fastica_fit, fastica_transform

    X = _data()
    kwargs = {
        "n_components": 2,
        "algorithm": "deflation",
        "whiten": "arbitrary-variance",
        "fun": "exp",
        "random_state": 0,
        "max_iter": 500,
        "tol": 1e-5,
    }
    state = fastica_fit(X, **kwargs)
    expected = SklearnFastICA(**kwargs).fit(X)

    assert np.allclose(state.components, expected.components_)
    assert np.allclose(state.mixing, expected.mixing_)
    assert np.allclose(fastica_transform(X, state), expected.transform(X))
    assert state.n_iter == expected.n_iter_


def test_fastica_eigh_and_cube_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.decomposition import fastica_fit, fastica_transform

    X = _data()
    kwargs = {
        "n_components": 2,
        "fun": "cube",
        "whiten_solver": "eigh",
        "random_state": 1,
        "max_iter": 500,
        "tol": 1e-5,
    }
    state = fastica_fit(X, **kwargs)
    expected = SklearnFastICA(**kwargs).fit(X)

    assert np.allclose(state.components, expected.components_)
    assert np.allclose(fastica_transform(X, state), expected.transform(X))
    assert state.n_iter == expected.n_iter_


def test_fastica_rejects_unsupported_options() -> None:
    from sciona.atoms.ml.sklearn.decomposition import fastica_fit

    X = _data()
    with pytest.raises(Exception):
        fastica_fit(X, fun="callable-placeholder")

    with pytest.raises(Exception):
        fastica_fit(X, fun_args={"alpha": 2.5})

    with pytest.raises(Exception):
        fastica_fit(X, n_components=X.shape[1] + 1)

    with pytest.raises(Exception):
        fastica_fit(X, w_init=np.ones((3, 3)), n_components=2)
