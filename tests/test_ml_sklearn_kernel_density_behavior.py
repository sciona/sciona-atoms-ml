from __future__ import annotations

import numpy as np
import pytest
from sklearn.neighbors import KernelDensity as SklearnKernelDensity


def _density_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(0)
    X = rng.randn(24, 3).astype(np.float64)
    query = rng.randn(8, 3).astype(np.float64)
    return X, query


def test_kernel_density_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.neighbors import (
        KernelDensityState,
        kernel_density_fit,
        kernel_density_sample,
        kernel_density_score,
        kernel_density_score_samples,
    )

    assert KernelDensityState is not None
    assert callable(kernel_density_fit)
    assert callable(kernel_density_score_samples)
    assert callable(kernel_density_score)
    assert callable(kernel_density_sample)


def test_kernel_density_score_samples_matches_sklearn_for_all_kernels() -> None:
    from sciona.atoms.ml.sklearn.neighbors import kernel_density_fit, kernel_density_score_samples

    X, query = _density_data()
    for kernel in ("gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"):
        state = kernel_density_fit(X, bandwidth=0.8, kernel=kernel)
        expected = SklearnKernelDensity(bandwidth=0.8, kernel=kernel).fit(X)
        assert np.allclose(kernel_density_score_samples(query, state), expected.score_samples(query))


def test_kernel_density_score_and_rule_bandwidth_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.neighbors import kernel_density_fit, kernel_density_score

    X, query = _density_data()
    weights = np.linspace(1.0, 2.0, X.shape[0], dtype=np.float64)
    for bandwidth in ("scott", "silverman"):
        state = kernel_density_fit(X, bandwidth=bandwidth, kernel="gaussian", sample_weight=weights)
        expected = SklearnKernelDensity(bandwidth=bandwidth, kernel="gaussian").fit(X, sample_weight=weights)
        assert np.allclose(kernel_density_score(query, state), expected.score(query))


def test_kernel_density_sampling_matches_sklearn_gaussian_and_tophat() -> None:
    from sciona.atoms.ml.sklearn.neighbors import kernel_density_fit, kernel_density_sample

    X, _ = _density_data()
    for kernel in ("gaussian", "tophat"):
        state = kernel_density_fit(X, bandwidth=0.25, kernel=kernel)
        expected = SklearnKernelDensity(bandwidth=0.25, kernel=kernel).fit(X)
        assert np.allclose(kernel_density_sample(state, 6, random_state=42), expected.sample(6, random_state=42))


def test_kernel_density_rejects_out_of_scope_inputs() -> None:
    from sciona.atoms.ml.sklearn.neighbors import kernel_density_fit, kernel_density_sample

    X, _ = _density_data()
    with pytest.raises(Exception):
        kernel_density_fit(X, bandwidth=0.0)
    with pytest.raises(Exception):
        kernel_density_fit(X, metric="manhattan")
    with pytest.raises(Exception):
        kernel_density_fit(X, sample_weight=np.zeros(X.shape[0], dtype=np.float64))
    state = kernel_density_fit(X, kernel="linear")
    with pytest.raises(Exception):
        kernel_density_sample(state, 2, random_state=0)
