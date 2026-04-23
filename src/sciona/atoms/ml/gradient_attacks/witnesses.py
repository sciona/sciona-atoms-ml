"""Ghost witnesses for gradient attack atoms.

Each witness mirrors the atom's interface using abstract types and captures
the semantic shape of the computation without executing it.
"""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_momentum_gradient_accumulation(
    gradient: AbstractArray,
    previous_accumulated: AbstractArray,
    momentum: float = 1.0,
) -> AbstractArray:
    """Ghost witness for MI-FGSM momentum gradient accumulation.

    Takes two same-shape arrays and a scalar momentum, returns an array
    of the same shape. The output is the momentum-weighted sum of the
    previous accumulation and the L1-normalized current gradient.
    """
    return gradient


def witness_ensemble_logit_fusion(
    logits: list[AbstractArray],
    weights: AbstractArray,
) -> AbstractArray:
    """Ghost witness for ensemble logit fusion.

    Takes a list of same-shape logit arrays and a weight vector, returns
    a single array with the same shape as any individual logit array.
    """
    return logits[0]


def witness_rounded_clipped_perturbation_step(
    noise: AbstractArray,
    clip_range: int = 2,
) -> AbstractArray:
    """Ghost witness for rounded clipped perturbation step.

    Takes an array and a clip range, returns an integer-valued array of
    the same shape with values in [-clip_range, clip_range].
    """
    return noise


def witness_adaptive_epsilon_strategy(
    max_epsilon: float,
    threshold: float = 8.0,
) -> dict[str, float]:
    """Ghost witness for adaptive epsilon strategy.

    Takes a perturbation budget and threshold, returns a configuration
    dict with keys n_models, n_iterations, and step_size.
    """
    return {"n_models": 5.0, "n_iterations": 20.0, "step_size": 0.01}
