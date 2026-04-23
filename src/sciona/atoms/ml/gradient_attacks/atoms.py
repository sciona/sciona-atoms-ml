"""Framework-agnostic gradient attack primitives in pure numpy.

Implements core building blocks from the 1st-place solutions to the NIPS 2017
adversarial non-targeted and targeted attack competitions. The atoms decompose
the MI-FGSM attack loop into reusable pieces: momentum accumulation, ensemble
logit fusion, quantized perturbation steps, and budget-dependent configuration.

All computation is pure numpy -- no TensorFlow dependency. The gradient itself
is assumed to be provided by the caller from whatever autodiff framework is
in use.

Source: adversarial-nontarget-1st/attack_iter.py (Apache 2.0)
        adversarial-target-1st/target_attack.py (Apache 2.0)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_adaptive_epsilon_strategy,
    witness_ensemble_logit_fusion,
    witness_momentum_gradient_accumulation,
    witness_rounded_clipped_perturbation_step,
)


# ---------------------------------------------------------------------------
# Atom 1: MI-FGSM momentum gradient accumulation
# ---------------------------------------------------------------------------


@register_atom(witness_momentum_gradient_accumulation)
@icontract.require(
    lambda gradient, previous_accumulated: gradient.shape == previous_accumulated.shape,
    "gradient and previous_accumulated must have the same shape",
)
@icontract.require(lambda momentum: momentum >= 0.0, "momentum must be non-negative")
@icontract.ensure(
    lambda gradient, result: result.shape == gradient.shape,
    "output must preserve the input shape",
)
def momentum_gradient_accumulation(
    gradient: NDArray[np.float64],
    previous_accumulated: NDArray[np.float64],
    momentum: float = 1.0,
) -> NDArray[np.float64]:
    """Accumulate gradient with momentum using L1-normalized current gradient.

    Core step of MI-FGSM (Dong et al. 2018): the current gradient is
    L1-normalized by its mean absolute value before accumulation. This
    prevents any single iteration's gradient magnitude from dominating
    the running sum. With the default momentum of 1.0 (full momentum),
    the accumulated history has equal weight to the new gradient direction.

    Derived from attack_iter.py line 185:
        noise = noise / tf.reduce_mean(tf.abs(noise), ...)
        noise = momentum * grad + noise
    """
    normalized = gradient / (np.mean(np.abs(gradient)) + 1e-12)
    accumulated: NDArray[np.float64] = momentum * previous_accumulated + normalized
    return accumulated


# ---------------------------------------------------------------------------
# Atom 2: Ensemble logit fusion
# ---------------------------------------------------------------------------


@register_atom(witness_ensemble_logit_fusion)
@icontract.require(
    lambda logits, weights: len(logits) == len(weights),
    "number of logit arrays must match number of weights",
)
@icontract.require(lambda logits: len(logits) >= 1, "need at least one model")
@icontract.require(
    lambda weights: all(w >= 0.0 for w in weights),
    "all weights must be non-negative",
)
@icontract.ensure(
    lambda logits, result: result.shape == logits[0].shape,
    "fused output shape must match individual logit shape",
)
def ensemble_logit_fusion(
    logits: list[NDArray[np.float64]],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Weighted average of logit arrays from multiple models.

    Per-model weights allow asymmetric weighting -- adversarially-trained
    models are deliberately downweighted (e.g., 0.25) because their
    gradients overfit to that model's adversarial training distribution.
    Standard models receive weight 1.0.

    Derived from attack_iter.py line 170:
        logits = (logits_v3 + 0.25 * logits_adv_v3 + ...) / 7.25
    The explicit weighted average replaces the ad-hoc sum-and-divide.
    """
    logits_stack = np.stack(logits, axis=0)
    fused: NDArray[np.float64] = np.average(logits_stack, axis=0, weights=weights)
    return fused


# ---------------------------------------------------------------------------
# Atom 3: Rounded clipped perturbation step
# ---------------------------------------------------------------------------


@register_atom(witness_rounded_clipped_perturbation_step)
@icontract.require(lambda clip_range: clip_range >= 1, "clip_range must be at least 1")
@icontract.ensure(
    lambda noise, result: result.shape == noise.shape,
    "output must preserve the input shape",
)
@icontract.ensure(
    lambda clip_range, result: np.all(result >= -clip_range) and np.all(result <= clip_range),
    "all values must be within [-clip_range, clip_range]",
)
def rounded_clipped_perturbation_step(
    noise: NDArray[np.float64],
    clip_range: int = 2,
) -> NDArray[np.float64]:
    """Multi-level quantized perturbation step: round then clip.

    Produces integer values in {-k, ..., 0, ..., k} where k = clip_range.
    Unlike the binary sign() used in standard FGSM, this allows larger
    per-pixel moves where the gradient is confident and zero-step where
    the accumulated noise is near zero. This exploits the perturbation
    budget more efficiently for targeted attacks.

    Derived from target_attack.py line 159:
        x = x - alpha * tf.clip_by_value(tf.round(noise), -2, 2)
    """
    perturbation: NDArray[np.float64] = np.clip(
        np.round(noise), -clip_range, clip_range
    )
    return perturbation


# ---------------------------------------------------------------------------
# Atom 4: Adaptive epsilon strategy
# ---------------------------------------------------------------------------


@register_atom(witness_adaptive_epsilon_strategy)
@icontract.require(lambda max_epsilon: max_epsilon > 0.0, "max_epsilon must be positive")
@icontract.require(lambda threshold: threshold > 0.0, "threshold must be positive")
@icontract.ensure(
    lambda result: all(k in result for k in ("n_models", "n_iterations", "step_size")),
    "result must contain n_models, n_iterations, and step_size keys",
)
@icontract.ensure(
    lambda result: result["n_models"] >= 1 and result["n_iterations"] >= 1 and result["step_size"] > 0.0,
    "all configuration values must be positive",
)
def adaptive_epsilon_strategy(
    max_epsilon: float,
    threshold: float = 8.0,
) -> dict[str, float]:
    """Select attack configuration based on perturbation budget.

    Large budget (max_epsilon >= threshold): use more models and fewer
    iterations with larger steps. The extra budget tolerates coarser
    updates, so model diversity matters more than update precision.

    Small budget (max_epsilon < threshold): use fewer models and more
    iterations with finer steps. Under tight budget the perturbation
    must be more precise, so each step must be smaller and more
    iterations are needed to converge.

    Derived from target_attack.py lines 231-241:
        if FLAGS.max_epsilon >= 8:
            graph_large (5 models, 20 iters, eps/12 step)
        else:
            graph_small (2 models, 40 iters, eps/28 step)
    """
    eps_normalized = 2.0 * max_epsilon / 255.0

    if max_epsilon >= threshold:
        config: dict[str, float] = {
            "n_models": 5.0,
            "n_iterations": 20.0,
            "step_size": eps_normalized / 12.0,
        }
    else:
        config = {
            "n_models": 2.0,
            "n_iterations": 40.0,
            "step_size": eps_normalized / 28.0,
        }
    return config
