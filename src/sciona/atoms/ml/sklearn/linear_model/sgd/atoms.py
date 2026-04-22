"""Deterministic sklearn stochastic-gradient helper atoms."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_passive_aggressive_classifier_sgd_config,
    witness_passive_aggressive_regressor_sgd_config,
    witness_sgd_l1_ratio_or_zero,
    witness_sgd_learning_rate_value,
    witness_sgd_modified_huber_proba,
    witness_sgd_passive_aggressive_step_size,
)


_BASIC_LEARNING_RATES = {"constant", "adaptive", "invscaling", "optimal"}
_PA_LEARNING_RATES = {"pa1", "pa2"}
_PA_CLASSIFIER_LOSSES = {"hinge", "squared_hinge"}
_PA_REGRESSOR_LOSSES = {"epsilon_insensitive", "squared_epsilon_insensitive"}


def _finite_scalar(value: float) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)))


def _finite_positive(value: float) -> bool:
    return bool(_finite_scalar(value) and float(value) > 0.0)


def _finite_nonnegative(value: float) -> bool:
    return bool(_finite_scalar(value) and float(value) >= 0.0)


def _l1_ratio_valid(l1_ratio: float | None) -> bool:
    if l1_ratio is None:
        return True
    return bool(_finite_scalar(l1_ratio) and 0.0 <= float(l1_ratio) <= 1.0)


def _learning_rate_valid(learning_rate: str) -> bool:
    return learning_rate in _BASIC_LEARNING_RATES


def _pa_learning_rate_valid(learning_rate: str) -> bool:
    return learning_rate in _PA_LEARNING_RATES


def _optimal_params_valid(learning_rate: str, alpha: float, t0: float) -> bool:
    if learning_rate != "optimal":
        return True
    return bool(_finite_positive(alpha) and _finite_nonnegative(t0))


def _score_shape_valid(scores: NDArray[np.float64], binary: bool) -> bool:
    try:
        values = np.asarray(scores, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if not np.all(np.isfinite(values)):
        return False
    if binary:
        return bool(values.ndim == 1 and values.shape[0] >= 1)
    return bool(values.ndim == 2 and values.shape[0] >= 1 and values.shape[1] >= 2)


def _proba_result_valid(result: NDArray[np.float64], scores: NDArray[np.float64], binary: bool) -> bool:
    values = np.asarray(result, dtype=np.float64)
    source = np.asarray(scores, dtype=np.float64)
    expected_shape = (source.shape[0], 2) if binary else source.shape
    return bool(
        values.shape == expected_shape
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and np.all(values <= 1.0)
        and np.allclose(values.sum(axis=1), 1.0)
    )


def _pa_classifier_loss_valid(loss: str) -> bool:
    return loss in _PA_CLASSIFIER_LOSSES


def _pa_regressor_loss_valid(loss: str) -> bool:
    return loss in _PA_REGRESSOR_LOSSES


def _config_result_valid(result: tuple[str, str, float]) -> bool:
    delegated_loss, learning_rate, alpha = result
    return bool(isinstance(delegated_loss, str) and learning_rate in _PA_LEARNING_RATES and alpha == 1.0)


@register_atom(witness_sgd_l1_ratio_or_zero)
@icontract.require(lambda l1_ratio: _l1_ratio_valid(l1_ratio), "l1_ratio must be None or a finite fraction")
@icontract.ensure(lambda result: _finite_scalar(result) and 0.0 <= result <= 1.0, "returned l1_ratio must be a finite fraction")
def sgd_l1_ratio_or_zero(l1_ratio: float | None) -> float:
    """Return the l1-ratio value consumed by sklearn's SGD kernel."""
    if l1_ratio is None:
        return 0.0
    return float(l1_ratio)


@register_atom(witness_sgd_learning_rate_value)
@icontract.require(lambda learning_rate: _learning_rate_valid(learning_rate), "learning_rate must be a basic SGD schedule")
@icontract.require(lambda eta0: _finite_positive(eta0), "eta0 must be positive")
@icontract.require(lambda t: _finite_positive(t), "t must be positive")
@icontract.require(lambda power_t: _finite_scalar(power_t), "power_t must be finite")
@icontract.require(lambda learning_rate, alpha, t0: _optimal_params_valid(learning_rate, alpha, t0), "optimal schedule requires positive alpha and nonnegative t0")
@icontract.ensure(lambda result: _finite_positive(result), "learning-rate value must be finite and positive")
def sgd_learning_rate_value(
    learning_rate: str,
    eta0: float,
    *,
    alpha: float = 0.0001,
    t: float = 1.0,
    power_t: float = 0.5,
    t0: float = 0.0,
) -> float:
    """Evaluate a non-PA SGD learning-rate schedule at one step."""
    if learning_rate in ("constant", "adaptive"):
        return float(eta0)
    if learning_rate == "invscaling":
        return float(eta0) / float(t) ** float(power_t)
    return 1.0 / (float(alpha) * (float(t) + float(t0)))


@register_atom(witness_sgd_passive_aggressive_step_size)
@icontract.require(lambda learning_rate: _pa_learning_rate_valid(learning_rate), "learning_rate must be pa1 or pa2")
@icontract.require(lambda loss_value: _finite_nonnegative(loss_value), "loss_value must be nonnegative")
@icontract.require(lambda squared_norm: _finite_positive(squared_norm), "squared_norm must be positive")
@icontract.require(lambda eta0: _finite_positive(eta0), "eta0 must be positive")
@icontract.ensure(lambda result: _finite_nonnegative(result), "step size must be finite and nonnegative")
def sgd_passive_aggressive_step_size(
    loss_value: float,
    squared_norm: float,
    eta0: float,
    *,
    learning_rate: str,
) -> float:
    """Compute a PA-I or PA-II step size from loss and feature norm."""
    if learning_rate == "pa1":
        return min(float(eta0), float(loss_value) / float(squared_norm))
    return float(loss_value) / (float(squared_norm) + 1.0 / (2.0 * float(eta0)))


@register_atom(witness_sgd_modified_huber_proba)
@icontract.require(lambda scores, binary: _score_shape_valid(scores, binary), "scores must be finite with sklearn-compatible shape")
@icontract.ensure(lambda result, scores, binary: _proba_result_valid(result, scores, binary), "probabilities must be normalized and bounded")
def sgd_modified_huber_proba(scores: NDArray[np.float64], *, binary: bool) -> NDArray[np.float64]:
    """Normalize modified-Huber decision scores as sklearn probabilities."""
    values = np.asarray(scores, dtype=np.float64)
    if binary:
        positive = (np.clip(values, -1.0, 1.0) + 1.0) / 2.0
        return np.asarray(np.column_stack((1.0 - positive, positive)), dtype=np.float64)

    prob = (np.clip(values, -1.0, 1.0) + 1.0) / 2.0
    prob_sum = prob.sum(axis=1)
    all_zero = prob_sum == 0.0
    if np.any(all_zero):
        prob[all_zero, :] = 1.0
        prob_sum[all_zero] = prob.shape[1]
    return np.asarray(prob / prob_sum.reshape((prob.shape[0], -1)), dtype=np.float64)


@register_atom(witness_passive_aggressive_classifier_sgd_config)
@icontract.require(lambda loss: _pa_classifier_loss_valid(loss), "classifier loss must be a passive-aggressive loss")
@icontract.ensure(lambda result: _config_result_valid(result), "config must use PA learning rate and alpha one")
def passive_aggressive_classifier_sgd_config(loss: str) -> tuple[str, str, float]:
    """Map a classifier loss option to the lower-level training settings.

    The tuple contains the internal loss name, update schedule name, and
    alpha value that sklearn passes on before the compiled solver runs.
    """
    learning_rate = "pa1" if loss == "hinge" else "pa2"
    return "hinge", learning_rate, 1.0


@register_atom(witness_passive_aggressive_regressor_sgd_config)
@icontract.require(lambda loss: _pa_regressor_loss_valid(loss), "regressor loss must be a passive-aggressive loss")
@icontract.ensure(lambda result: _config_result_valid(result), "config must use PA learning rate and alpha one")
def passive_aggressive_regressor_sgd_config(loss: str) -> tuple[str, str, float]:
    """Map a regressor loss option to the lower-level training settings.

    The tuple contains the internal loss name, update schedule name, and
    alpha value that sklearn passes on before the compiled solver runs.
    """
    learning_rate = "pa1" if loss == "epsilon_insensitive" else "pa2"
    return "epsilon_insensitive", learning_rate, 1.0
