from __future__ import annotations

import numpy as np
import pytest
from sklearn.kernel_approximation import (
    AdditiveChi2Sampler as SklearnAdditiveChi2Sampler,
    SkewedChi2Sampler as SklearnSkewedChi2Sampler,
)


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


def test_chi2_sampler_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.kernel_approximation import (
        SkewedChi2SamplerState,
        additive_chi2_sampler_transform,
        skewed_chi2_sampler_fit,
        skewed_chi2_sampler_transform,
    )

    assert SkewedChi2SamplerState is not None
    assert callable(skewed_chi2_sampler_fit)
    assert callable(skewed_chi2_sampler_transform)
    assert callable(additive_chi2_sampler_transform)


def test_skewed_chi2_sampler_fit_and_transform_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.kernel_approximation import skewed_chi2_sampler_fit, skewed_chi2_sampler_transform

    X = _data()
    state = skewed_chi2_sampler_fit(X, skewedness=0.5, n_components=5, random_state=7)
    expected = SklearnSkewedChi2Sampler(skewedness=0.5, n_components=5, random_state=7).fit(X)

    assert np.allclose(state.random_weights, expected.random_weights_)
    assert np.allclose(state.random_offset, expected.random_offset_)
    assert state.n_features_in == expected.n_features_in_
    assert np.allclose(skewed_chi2_sampler_transform(X, state), expected.transform(X))


def test_additive_chi2_sampler_transform_matches_sklearn_defaults_and_custom_interval() -> None:
    from sciona.atoms.ml.sklearn.kernel_approximation import additive_chi2_sampler_transform

    X = _data()
    result = additive_chi2_sampler_transform(X, sample_steps=2)
    expected = SklearnAdditiveChi2Sampler(sample_steps=2).fit(X).transform(X)
    assert np.allclose(result, expected)

    custom = additive_chi2_sampler_transform(X, sample_steps=4, sample_interval=0.25)
    custom_expected = SklearnAdditiveChi2Sampler(sample_steps=4, sample_interval=0.25).fit(X).transform(X)
    assert np.allclose(custom, custom_expected)


def test_chi2_samplers_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.kernel_approximation import (
        additive_chi2_sampler_transform,
        skewed_chi2_sampler_fit,
        skewed_chi2_sampler_transform,
    )

    X = _data()
    with pytest.raises(Exception):
        skewed_chi2_sampler_fit(X, n_components=0)

    state = skewed_chi2_sampler_fit(X, skewedness=0.25, n_components=3, random_state=0)
    with pytest.raises(Exception):
        skewed_chi2_sampler_transform(np.array([[-0.25, 0.0, 1.0]], dtype=np.float64), state)

    with pytest.raises(Exception):
        additive_chi2_sampler_transform(np.array([[-1.0, 0.0]], dtype=np.float64))

    with pytest.raises(Exception):
        additive_chi2_sampler_transform(X, sample_steps=4)
