from __future__ import annotations

import numpy as np
import pytest


def test_decorrelation_atoms_import() -> None:
    from sciona.atoms.ml.constrained_ml.decorrelation.atoms import (
        compute_cvm_mass_decorrelation,
        compute_ks_agreement,
        noise_injection_decorrelation,
        roc_auc_truncated_weighted,
    )
    assert callable(compute_cvm_mass_decorrelation)
    assert callable(compute_ks_agreement)
    assert callable(roc_auc_truncated_weighted)
    assert callable(noise_injection_decorrelation)


# ---------------------------------------------------------------------------
# CvM mass decorrelation
# ---------------------------------------------------------------------------


def test_cvm_independent_predictions_low() -> None:
    """When predictions are independent of mass, CvM should be near zero."""
    from sciona.atoms.ml.constrained_ml.decorrelation.atoms import (
        compute_cvm_mass_decorrelation,
    )

    rng = np.random.default_rng(42)
    n = 2000
    predictions = rng.uniform(0, 1, size=n)
    mass = rng.uniform(1.0, 5.0, size=n)
    cvm = compute_cvm_mass_decorrelation(predictions, mass, n_neighbours=200, step=50)
    # Should be small for independent variables
    assert cvm < 0.01


def test_cvm_correlated_predictions_higher() -> None:
    """When predictions are correlated with mass, CvM should be larger."""
    from sciona.atoms.ml.constrained_ml.decorrelation.atoms import (
        compute_cvm_mass_decorrelation,
    )

    rng = np.random.default_rng(42)
    n = 2000
    mass = rng.uniform(1.0, 5.0, size=n)
    # predictions perfectly correlated with mass
    predictions = mass / 5.0
    cvm = compute_cvm_mass_decorrelation(predictions, mass, n_neighbours=200, step=50)
    # Should be noticeably larger than the independent case
    assert cvm > 0.01


def test_cvm_small_dataset() -> None:
    """CvM should work on small datasets by clamping window size."""
    from sciona.atoms.ml.constrained_ml.decorrelation.atoms import (
        compute_cvm_mass_decorrelation,
    )

    rng = np.random.default_rng(0)
    predictions = rng.uniform(0, 1, size=20)
    mass = rng.uniform(0, 1, size=20)
    cvm = compute_cvm_mass_decorrelation(predictions, mass, n_neighbours=200, step=5)
    assert np.isfinite(cvm) and cvm >= 0.0


# ---------------------------------------------------------------------------
# KS agreement
# ---------------------------------------------------------------------------


def test_ks_identical_distributions_near_zero() -> None:
    """KS distance between two samples from the same distribution should be small."""
    from sciona.atoms.ml.constrained_ml.decorrelation.atoms import (
        compute_ks_agreement,
    )

    rng = np.random.default_rng(42)
    n = 1000
    data = rng.uniform(0, 1, size=n)
    mc = rng.uniform(0, 1, size=n)
    w = np.ones(n)
    ks = compute_ks_agreement(data, mc, w, w)
    assert ks < 0.1


def test_ks_different_distributions_high() -> None:
    """KS distance between very different distributions should be high."""
    from sciona.atoms.ml.constrained_ml.decorrelation.atoms import (
        compute_ks_agreement,
    )

    data = np.zeros(100)
    mc = np.ones(100)
    w = np.ones(100)
    ks = compute_ks_agreement(data, mc, w, w)
    assert ks > 0.5


def test_ks_weighted() -> None:
    """KS should respect sample weights."""
    from sciona.atoms.ml.constrained_ml.decorrelation.atoms import (
        compute_ks_agreement,
    )

    rng = np.random.default_rng(42)
    data = rng.uniform(0, 1, size=200)
    mc = rng.uniform(0, 1, size=200)
    w_uniform = np.ones(200)
    w_heavy = rng.uniform(0.1, 10.0, size=200)
    # Both should be valid and finite
    ks1 = compute_ks_agreement(data, mc, w_uniform, w_uniform)
    ks2 = compute_ks_agreement(data, mc, w_heavy, w_heavy)
    assert 0.0 <= ks1 <= 1.0
    assert 0.0 <= ks2 <= 1.0


# ---------------------------------------------------------------------------
# Truncated weighted AUC
# ---------------------------------------------------------------------------


def test_truncated_auc_perfect_classifier() -> None:
    """A perfect classifier should get AUC close to 1.0."""
    from sciona.atoms.ml.constrained_ml.decorrelation.atoms import (
        roc_auc_truncated_weighted,
    )

    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.float64)
    predictions = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float64)
    auc = roc_auc_truncated_weighted(labels, predictions)
    assert auc > 0.9


def test_truncated_auc_random_classifier_around_half() -> None:
    """A random classifier should get AUC near 0.5."""
    from sciona.atoms.ml.constrained_ml.decorrelation.atoms import (
        roc_auc_truncated_weighted,
    )

    rng = np.random.default_rng(42)
    n = 5000
    labels = rng.integers(0, 2, size=n).astype(np.float64)
    predictions = rng.uniform(0, 1, size=n)
    auc = roc_auc_truncated_weighted(labels, predictions)
    assert 0.2 < auc < 0.8


def test_truncated_auc_custom_weights() -> None:
    """Custom weights should work without error."""
    from sciona.atoms.ml.constrained_ml.decorrelation.atoms import (
        roc_auc_truncated_weighted,
    )

    rng = np.random.default_rng(42)
    labels = rng.integers(0, 2, size=200).astype(np.float64)
    predictions = rng.uniform(0, 1, size=200)
    auc = roc_auc_truncated_weighted(
        labels, predictions,
        tpr_thresholds=(0.5,),
        weights=(1.0, 1.0),
    )
    assert 0.0 <= auc <= 1.0


# ---------------------------------------------------------------------------
# Noise injection decorrelation
# ---------------------------------------------------------------------------


def test_noise_injection_zero_level_identity() -> None:
    """With noise_level=0, output should equal input."""
    from sciona.atoms.ml.constrained_ml.decorrelation.atoms import (
        noise_injection_decorrelation,
    )

    predictions = np.array([0.1, 0.5, 0.9], dtype=np.float64)
    result = noise_injection_decorrelation(predictions, noise_level=0.0, random_state=42)
    np.testing.assert_allclose(result, predictions)


def test_noise_injection_full_level_uniform() -> None:
    """With noise_level=1, output should be pure noise (uniform-distributed)."""
    from sciona.atoms.ml.constrained_ml.decorrelation.atoms import (
        noise_injection_decorrelation,
    )

    rng = np.random.default_rng(42)
    predictions = rng.uniform(0, 1, size=5000)
    result = noise_injection_decorrelation(predictions, noise_level=1.0, random_state=99)
    # Pure noise: should still be in [0,1] and independent of input
    assert np.all(result >= 0.0) and np.all(result <= 1.0)


def test_noise_injection_shape_preserved() -> None:
    """Output shape must match input shape."""
    from sciona.atoms.ml.constrained_ml.decorrelation.atoms import (
        noise_injection_decorrelation,
    )

    predictions = np.linspace(0, 1, 50, dtype=np.float64)
    result = noise_injection_decorrelation(predictions, noise_level=0.3, random_state=0)
    assert result.shape == predictions.shape


def test_noise_injection_reproducible() -> None:
    """Same random_state should produce identical results."""
    from sciona.atoms.ml.constrained_ml.decorrelation.atoms import (
        noise_injection_decorrelation,
    )

    predictions = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float64)
    r1 = noise_injection_decorrelation(predictions, noise_level=0.5, random_state=42)
    r2 = noise_injection_decorrelation(predictions, noise_level=0.5, random_state=42)
    np.testing.assert_array_equal(r1, r2)
