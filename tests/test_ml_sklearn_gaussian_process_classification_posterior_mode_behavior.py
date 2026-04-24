from __future__ import annotations

import numpy as np
import pytest
from sklearn.gaussian_process._gpc import _BinaryGaussianProcessClassifierLaplace


def test_gp_classifier_posterior_mode_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_posterior_mode import (
        gp_classifier_posterior_mode,
        gp_classifier_posterior_mode_converged,
        gp_classifier_posterior_mode_initial_latent,
    )

    assert callable(gp_classifier_posterior_mode_initial_latent)
    assert callable(gp_classifier_posterior_mode_converged)
    assert callable(gp_classifier_posterior_mode)


def _kernel_and_targets() -> tuple[np.ndarray, np.ndarray]:
    K = np.array(
        [
            [1.6, 0.2, 0.1],
            [0.2, 1.3, 0.25],
            [0.1, 0.25, 1.4],
        ],
        dtype=np.float64,
    )
    y_train = np.array([0.0, 1.0, 1.0], dtype=np.float64)
    return K, y_train


def test_gp_classifier_posterior_mode_initial_latent_matches_sklearn_zero_init() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_posterior_mode import (
        gp_classifier_posterior_mode_initial_latent,
    )

    _, y_train = _kernel_and_targets()
    result = gp_classifier_posterior_mode_initial_latent(y_train, warm_start=False)
    assert np.array_equal(result, np.zeros_like(y_train, dtype=np.float64))


def test_gp_classifier_posterior_mode_initial_latent_matches_sklearn_warm_start_choice() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_posterior_mode import (
        gp_classifier_posterior_mode_initial_latent,
    )

    _, y_train = _kernel_and_targets()
    cached = np.array([0.4, -0.2, 0.1], dtype=np.float64)
    assert np.array_equal(
        gp_classifier_posterior_mode_initial_latent(y_train, warm_start=True, cached_f=cached),
        cached,
    )
    assert np.array_equal(
        gp_classifier_posterior_mode_initial_latent(y_train, warm_start=True, cached_f=np.array([1.0, 2.0], dtype=np.float64)),
        np.zeros_like(y_train, dtype=np.float64),
    )


def test_gp_classifier_posterior_mode_converged_matches_sklearn_rule() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_posterior_mode import (
        gp_classifier_posterior_mode_converged,
    )

    assert gp_classifier_posterior_mode_converged(float("-inf"), -5.0) is False
    assert gp_classifier_posterior_mode_converged(-3.0, -2.99999999995) is True
    assert gp_classifier_posterior_mode_converged(-3.0, -2.9) is False


def test_gp_classifier_posterior_mode_matches_sklearn_without_warm_start() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_posterior_mode import (
        gp_classifier_posterior_mode,
    )

    K, y_train = _kernel_and_targets()
    model = _BinaryGaussianProcessClassifierLaplace(max_iter_predict=7, warm_start=False)
    model.y_train_ = y_train.astype(np.int64)

    expected_lml, (expected_pi, expected_w_sr, expected_L, expected_b, expected_a) = model._posterior_mode(
        K,
        return_temporaries=True,
    )
    result = gp_classifier_posterior_mode(
        K,
        y_train,
        max_iter_predict=7,
        warm_start=False,
    )

    lml, f_cached, pi, w_sr, L, b, a = result
    assert lml == pytest.approx(expected_lml)
    assert np.allclose(f_cached, model.f_cached)
    assert np.allclose(pi, expected_pi)
    assert np.allclose(w_sr, expected_w_sr)
    assert np.allclose(L, expected_L)
    assert np.allclose(b, expected_b)
    assert np.allclose(a, expected_a)


def test_gp_classifier_posterior_mode_matches_sklearn_with_warm_start() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_posterior_mode import (
        gp_classifier_posterior_mode,
    )

    K, y_train = _kernel_and_targets()
    cached = np.array([0.2, -0.1, 0.05], dtype=np.float64)
    model = _BinaryGaussianProcessClassifierLaplace(max_iter_predict=6, warm_start=True)
    model.y_train_ = y_train.astype(np.int64)
    model.f_cached = cached.copy()

    expected_lml, (expected_pi, expected_w_sr, expected_L, expected_b, expected_a) = model._posterior_mode(
        K,
        return_temporaries=True,
    )
    result = gp_classifier_posterior_mode(
        K,
        y_train,
        max_iter_predict=6,
        warm_start=True,
        cached_f=cached,
    )

    lml, f_cached, pi, w_sr, L, b, a = result
    assert lml == pytest.approx(expected_lml)
    assert np.allclose(f_cached, model.f_cached)
    assert np.allclose(pi, expected_pi)
    assert np.allclose(w_sr, expected_w_sr)
    assert np.allclose(L, expected_L)
    assert np.allclose(b, expected_b)
    assert np.allclose(a, expected_a)


def test_gp_classifier_posterior_mode_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_posterior_mode import (
        gp_classifier_posterior_mode,
        gp_classifier_posterior_mode_converged,
        gp_classifier_posterior_mode_initial_latent,
    )

    K, y_train = _kernel_and_targets()
    with pytest.raises(Exception):
        gp_classifier_posterior_mode_initial_latent(np.array([0.0, 2.0], dtype=np.float64))
    with pytest.raises(Exception):
        gp_classifier_posterior_mode_converged(-1.0, -0.5, tolerance=0.0)
    with pytest.raises(Exception):
        gp_classifier_posterior_mode(K, y_train, max_iter_predict=0)
    with pytest.raises(Exception):
        gp_classifier_posterior_mode(K, np.array([0.0, 1.0], dtype=np.float64))
