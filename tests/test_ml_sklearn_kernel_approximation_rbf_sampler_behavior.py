from __future__ import annotations

import numpy as np
import pytest
from sklearn.kernel_approximation import RBFSampler as SklearnRBFSampler


def _data() -> np.ndarray:
    return np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 0.0],
            [3.0, 4.0, 1.0],
        ],
        dtype=np.float64,
    )


def test_rbf_sampler_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.kernel_approximation import RBFSamplerState, rbf_sampler_fit, rbf_sampler_transform

    assert RBFSamplerState is not None
    assert callable(rbf_sampler_fit)
    assert callable(rbf_sampler_transform)


def test_rbf_sampler_fit_and_transform_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.kernel_approximation import rbf_sampler_fit, rbf_sampler_transform

    X = _data()
    state = rbf_sampler_fit(X, gamma=0.75, n_components=5, random_state=13)
    expected = SklearnRBFSampler(gamma=0.75, n_components=5, random_state=13).fit(X)

    assert np.allclose(state.random_weights, expected.random_weights_)
    assert np.allclose(state.random_offset, expected.random_offset_)
    assert state.gamma == expected._gamma
    assert state.n_features_in == expected.n_features_in_
    assert np.allclose(rbf_sampler_transform(X, state), expected.transform(X))


def test_rbf_sampler_scale_gamma_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.kernel_approximation import rbf_sampler_fit, rbf_sampler_transform

    X = _data()
    state = rbf_sampler_fit(X, gamma="scale", n_components=4, random_state=3)
    expected = SklearnRBFSampler(gamma="scale", n_components=4, random_state=3).fit(X)

    assert np.isclose(state.gamma, expected._gamma)
    assert np.allclose(state.random_weights, expected.random_weights_)
    assert np.allclose(rbf_sampler_transform(X[:2], state), expected.transform(X[:2]))


def test_rbf_sampler_rejects_bad_inputs() -> None:
    from sciona.atoms.ml.sklearn.kernel_approximation import rbf_sampler_fit, rbf_sampler_transform

    X = _data()
    with pytest.raises(Exception):
        rbf_sampler_fit(X, gamma=0.0)

    with pytest.raises(Exception):
        rbf_sampler_fit(X, n_components=0)

    state = rbf_sampler_fit(X, n_components=3, random_state=0)
    with pytest.raises(Exception):
        rbf_sampler_transform(np.ones((2, 2), dtype=np.float64), state)
