"""Ghost witnesses for constrained ML decorrelation atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_compute_cvm_mass_decorrelation(
    predictions: AbstractArray,
    protected_variable: AbstractArray,
    n_neighbours: int = 200,
    step: int = 50,
) -> float:
    """CvM decorrelation statistic is a non-negative scalar."""
    return 0.0


def witness_compute_ks_agreement(
    data_predictions: AbstractArray,
    mc_predictions: AbstractArray,
    weights_data: AbstractArray,
    weights_mc: AbstractArray,
) -> float:
    """KS distance is a scalar in [0, 1]."""
    return 0.0


def witness_roc_auc_truncated_weighted(
    labels: AbstractArray,
    predictions: AbstractArray,
    tpr_thresholds: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8),
    weights: tuple[float, ...] = (4, 3, 2, 1, 0),
) -> float:
    """Weighted truncated AUC is a scalar in [0, 1]."""
    return 0.5


def witness_noise_injection_decorrelation(
    predictions: AbstractArray,
    noise_level: float = 0.40,
    random_state: int | None = None,
) -> AbstractArray:
    """Noise-injected predictions have the same shape as input."""
    return AbstractArray(shape=predictions.shape, dtype="float64")
