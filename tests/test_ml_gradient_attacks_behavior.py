from __future__ import annotations

import numpy as np
import pytest


def test_gradient_attacks_import() -> None:
    from sciona.atoms.ml.gradient_attacks.atoms import (
        adaptive_epsilon_strategy,
        ensemble_logit_fusion,
        momentum_gradient_accumulation,
        rounded_clipped_perturbation_step,
    )
    assert callable(momentum_gradient_accumulation)
    assert callable(ensemble_logit_fusion)
    assert callable(rounded_clipped_perturbation_step)
    assert callable(adaptive_epsilon_strategy)


# ---------------------------------------------------------------------------
# momentum_gradient_accumulation
# ---------------------------------------------------------------------------


def test_momentum_accumulation_shape_preserved() -> None:
    from sciona.atoms.ml.gradient_attacks.atoms import momentum_gradient_accumulation

    grad = np.random.default_rng(42).standard_normal((3, 4))
    prev = np.zeros_like(grad)
    result = momentum_gradient_accumulation(grad, prev, momentum=1.0)
    assert result.shape == grad.shape


def test_momentum_accumulation_zero_momentum_ignores_history() -> None:
    from sciona.atoms.ml.gradient_attacks.atoms import momentum_gradient_accumulation

    grad = np.ones((2, 3), dtype=np.float64)
    prev = np.full((2, 3), 100.0, dtype=np.float64)
    result = momentum_gradient_accumulation(grad, prev, momentum=0.0)
    # With zero momentum, previous should be ignored; result is just normalized grad
    expected = grad / (np.mean(np.abs(grad)) + 1e-12)
    np.testing.assert_allclose(result, expected)


def test_momentum_accumulation_full_momentum_adds_history() -> None:
    from sciona.atoms.ml.gradient_attacks.atoms import momentum_gradient_accumulation

    grad = np.ones((2, 2), dtype=np.float64)
    prev = np.ones((2, 2), dtype=np.float64) * 2.0
    result = momentum_gradient_accumulation(grad, prev, momentum=1.0)
    normalized = grad / (np.mean(np.abs(grad)) + 1e-12)
    expected = 1.0 * prev + normalized
    np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# ensemble_logit_fusion
# ---------------------------------------------------------------------------


def test_ensemble_logit_fusion_uniform_weights() -> None:
    from sciona.atoms.ml.gradient_attacks.atoms import ensemble_logit_fusion

    logits_a = np.array([1.0, 2.0, 3.0])
    logits_b = np.array([3.0, 2.0, 1.0])
    weights = np.array([1.0, 1.0])
    result = ensemble_logit_fusion([logits_a, logits_b], weights)
    np.testing.assert_allclose(result, [2.0, 2.0, 2.0])


def test_ensemble_logit_fusion_asymmetric_weights() -> None:
    from sciona.atoms.ml.gradient_attacks.atoms import ensemble_logit_fusion

    logits_a = np.array([10.0, 0.0])
    logits_b = np.array([0.0, 10.0])
    weights = np.array([1.0, 0.25])
    result = ensemble_logit_fusion([logits_a, logits_b], weights)
    # np.average with weights [1.0, 0.25] => (1.0*10 + 0.25*0) / 1.25 = 8.0
    expected = np.average(np.stack([logits_a, logits_b]), axis=0, weights=weights)
    np.testing.assert_allclose(result, expected)


def test_ensemble_logit_fusion_preserves_shape() -> None:
    from sciona.atoms.ml.gradient_attacks.atoms import ensemble_logit_fusion

    rng = np.random.default_rng(42)
    logits = [rng.standard_normal((5, 1001)) for _ in range(3)]
    weights = np.array([1.0, 0.25, 1.0])
    result = ensemble_logit_fusion(logits, weights)
    assert result.shape == (5, 1001)


# ---------------------------------------------------------------------------
# rounded_clipped_perturbation_step
# ---------------------------------------------------------------------------


def test_rounded_clipped_output_is_integer_valued() -> None:
    from sciona.atoms.ml.gradient_attacks.atoms import rounded_clipped_perturbation_step

    noise = np.array([0.3, -1.7, 2.5, -0.1, 3.9])
    result = rounded_clipped_perturbation_step(noise, clip_range=2)
    # All values should be integers
    np.testing.assert_array_equal(result, np.round(result))


def test_rounded_clipped_respects_clip_range() -> None:
    from sciona.atoms.ml.gradient_attacks.atoms import rounded_clipped_perturbation_step

    noise = np.array([10.0, -10.0, 0.0, 1.4, -1.6])
    result = rounded_clipped_perturbation_step(noise, clip_range=2)
    assert np.all(result >= -2)
    assert np.all(result <= 2)
    np.testing.assert_allclose(result, [2.0, -2.0, 0.0, 1.0, -2.0])


def test_rounded_clipped_near_zero_stays_zero() -> None:
    from sciona.atoms.ml.gradient_attacks.atoms import rounded_clipped_perturbation_step

    noise = np.array([0.1, -0.1, 0.49, -0.49])
    result = rounded_clipped_perturbation_step(noise, clip_range=2)
    np.testing.assert_allclose(result, [0.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# adaptive_epsilon_strategy
# ---------------------------------------------------------------------------


def test_adaptive_strategy_large_epsilon() -> None:
    from sciona.atoms.ml.gradient_attacks.atoms import adaptive_epsilon_strategy

    config = adaptive_epsilon_strategy(max_epsilon=16.0)
    assert config["n_models"] == 5.0
    assert config["n_iterations"] == 20.0
    eps_norm = 2.0 * 16.0 / 255.0
    assert config["step_size"] == pytest.approx(eps_norm / 12.0)


def test_adaptive_strategy_small_epsilon() -> None:
    from sciona.atoms.ml.gradient_attacks.atoms import adaptive_epsilon_strategy

    config = adaptive_epsilon_strategy(max_epsilon=4.0)
    assert config["n_models"] == 2.0
    assert config["n_iterations"] == 40.0
    eps_norm = 2.0 * 4.0 / 255.0
    assert config["step_size"] == pytest.approx(eps_norm / 28.0)


def test_adaptive_strategy_boundary_at_threshold() -> None:
    from sciona.atoms.ml.gradient_attacks.atoms import adaptive_epsilon_strategy

    # Exactly at threshold should use large config
    config = adaptive_epsilon_strategy(max_epsilon=8.0)
    assert config["n_models"] == 5.0


def test_adaptive_strategy_custom_threshold() -> None:
    from sciona.atoms.ml.gradient_attacks.atoms import adaptive_epsilon_strategy

    config = adaptive_epsilon_strategy(max_epsilon=10.0, threshold=12.0)
    # 10 < 12, so should use small config
    assert config["n_models"] == 2.0
    assert config["n_iterations"] == 40.0
