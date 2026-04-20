from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import FactorAnalysis as SklearnFactorAnalysis


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


def test_factor_analysis_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.decomposition import (
        FactorAnalysisState,
        factor_analysis_covariance,
        factor_analysis_fit,
        factor_analysis_precision,
        factor_analysis_score,
        factor_analysis_score_samples,
        factor_analysis_transform,
    )

    assert FactorAnalysisState is not None
    assert callable(factor_analysis_fit)
    assert callable(factor_analysis_transform)
    assert callable(factor_analysis_covariance)
    assert callable(factor_analysis_precision)
    assert callable(factor_analysis_score_samples)
    assert callable(factor_analysis_score)


def test_factor_analysis_fit_and_methods_match_sklearn_lapack() -> None:
    from sciona.atoms.ml.sklearn.decomposition import (
        factor_analysis_covariance,
        factor_analysis_fit,
        factor_analysis_precision,
        factor_analysis_score,
        factor_analysis_score_samples,
        factor_analysis_transform,
    )

    X = _data()
    state = factor_analysis_fit(X, n_components=2, max_iter=200, tol=1e-4)
    expected = SklearnFactorAnalysis(n_components=2, svd_method="lapack", rotation=None, max_iter=200, tol=1e-4).fit(X)

    assert np.allclose(state.components, expected.components_)
    assert np.allclose(state.noise_variance, expected.noise_variance_)
    assert np.allclose(state.mean, expected.mean_)
    assert np.allclose(state.loglike, np.asarray(expected.loglike_))
    assert state.n_iter == expected.n_iter_
    assert state.n_features_in == expected.n_features_in_

    query = X[[0, 2, 5]]
    assert np.allclose(factor_analysis_transform(query, state), expected.transform(query))
    assert np.allclose(factor_analysis_covariance(state), expected.get_covariance())
    assert np.allclose(factor_analysis_precision(state), expected.get_precision())
    assert np.allclose(factor_analysis_score_samples(query, state), expected.score_samples(query))
    assert np.isclose(factor_analysis_score(X, state), expected.score(X))


def test_factor_analysis_default_components_and_noise_init_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.decomposition import factor_analysis_fit

    X = _data()
    noise = (0.8, 1.2, 1.1, 0.9)
    state = factor_analysis_fit(X, noise_variance_init=noise, max_iter=20)
    expected = SklearnFactorAnalysis(
        svd_method="lapack",
        rotation=None,
        noise_variance_init=np.asarray(noise),
        max_iter=20,
    ).fit(X)

    assert state.n_components == X.shape[1]
    assert np.allclose(state.components, expected.components_)
    assert np.allclose(state.noise_variance, expected.noise_variance_)


def test_factor_analysis_zero_components_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.decomposition import factor_analysis_fit, factor_analysis_transform

    X = _data()
    state = factor_analysis_fit(X, n_components=0, max_iter=20)
    expected = SklearnFactorAnalysis(n_components=0, svd_method="lapack", rotation=None, max_iter=20).fit(X)

    assert state.components.shape == expected.components_.shape
    assert np.allclose(state.noise_variance, expected.noise_variance_)
    transformed = factor_analysis_transform(X[:3], state)
    assert transformed.shape == (3, 0)
    assert np.allclose(transformed, expected.transform(X[:3]))


def test_factor_analysis_rejects_unsupported_options() -> None:
    from sciona.atoms.ml.sklearn.decomposition import factor_analysis_fit

    X = _data()
    with pytest.raises(Exception):
        factor_analysis_fit(X, svd_method="randomized")

    with pytest.raises(Exception):
        factor_analysis_fit(X, rotation="varimax")

    with pytest.raises(Exception):
        factor_analysis_fit(X, n_components=X.shape[1] + 1)

    with pytest.raises(Exception):
        factor_analysis_fit(X, noise_variance_init=(1.0, 1.0))

    with pytest.raises(Exception):
        factor_analysis_fit(np.ones((4, 3), dtype=np.float64))
