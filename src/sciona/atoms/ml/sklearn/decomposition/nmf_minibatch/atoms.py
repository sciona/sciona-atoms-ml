"""MiniBatchNMF scheduling and convergence atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_nmf_minibatch_batch_size,
    witness_nmf_minibatch_ewa_cost,
    witness_nmf_minibatch_h_change_converged,
    witness_nmf_minibatch_improvement_state,
    witness_nmf_minibatch_mm_gamma,
    witness_nmf_minibatch_rho,
    witness_nmf_minibatch_transform_max_iter,
)


def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _nonnegative_int_or_none(value: int | None) -> bool:
    return value is None or (isinstance(value, int) and not isinstance(value, bool) and value >= 0)


def _positive_int_or_none(value: int | None) -> bool:
    return value is None or _positive_int(value)


def _finite_scalar(value: float | int) -> bool:
    return bool(not isinstance(value, bool) and np.isscalar(value) and np.isfinite(float(value)))


def _nonnegative_finite_scalar(value: float | int) -> bool:
    return bool(_finite_scalar(value) and float(value) >= 0.0)


def _unit_interval(value: float | int) -> bool:
    return bool(_finite_scalar(value) and 0.0 <= float(value) <= 1.0)


def _finite_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _same_shape(X: object, Y: object) -> bool:
    return bool(_finite_matrix(X) and _finite_matrix(Y) and np.asarray(X).shape == np.asarray(Y).shape)


@register_atom(witness_nmf_minibatch_batch_size)
@icontract.require(lambda batch_size: _positive_int(batch_size), "batch_size must be a positive integer")
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be a positive integer")
@icontract.ensure(lambda result, n_samples: _positive_int(result) and result <= n_samples, "resolved batch size must be a positive integer not exceeding n_samples")
def nmf_minibatch_batch_size(batch_size: int, n_samples: int) -> int:
    """Clamp MiniBatchNMF batch size to the available sample count."""
    return int(min(batch_size, n_samples))


@register_atom(witness_nmf_minibatch_rho)
@icontract.require(lambda forget_factor: _unit_interval(forget_factor), "forget_factor must lie in [0, 1]")
@icontract.require(lambda batch_size: _positive_int(batch_size), "batch_size must be a positive integer")
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be a positive integer")
@icontract.ensure(lambda result: _nonnegative_finite_scalar(result), "rho must be finite and nonnegative")
def nmf_minibatch_rho(forget_factor: float, batch_size: int, n_samples: int) -> float:
    """Compute sklearn's forgetting-rate power for one minibatch configuration."""
    return float(float(forget_factor) ** (batch_size / n_samples))


@register_atom(witness_nmf_minibatch_mm_gamma)
@icontract.require(lambda beta_loss: _finite_scalar(beta_loss), "beta_loss must be finite")
@icontract.ensure(lambda result: _nonnegative_finite_scalar(result) and float(result) > 0.0, "MM gamma must be finite and positive")
def nmf_minibatch_mm_gamma(beta_loss: float) -> float:
    """Compute the MM gamma selected by sklearn from beta loss."""
    beta_value = float(beta_loss)
    if beta_value < 1.0:
        return float(1.0 / (2.0 - beta_value))
    if beta_value > 2.0:
        return float(1.0 / (beta_value - 1.0))
    return 1.0


@register_atom(witness_nmf_minibatch_transform_max_iter)
@icontract.require(lambda max_iter: _positive_int(max_iter), "max_iter must be a positive integer")
@icontract.require(lambda transform_max_iter: _positive_int_or_none(transform_max_iter), "transform_max_iter must be None or a positive integer")
@icontract.ensure(lambda result: _positive_int(result), "resolved transform_max_iter must be a positive integer")
def nmf_minibatch_transform_max_iter(max_iter: int, transform_max_iter: int | None) -> int:
    """Resolve sklearn's transform iteration cap for MiniBatchNMF."""
    return int(max_iter if transform_max_iter is None else transform_max_iter)


@register_atom(witness_nmf_minibatch_ewa_cost)
@icontract.require(lambda batch_cost: _finite_scalar(batch_cost), "batch_cost must be finite")
@icontract.require(lambda previous_ewa_cost: previous_ewa_cost is None or _finite_scalar(previous_ewa_cost), "previous_ewa_cost must be None or finite")
@icontract.require(lambda batch_size: _positive_int(batch_size), "batch_size must be a positive integer")
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be a positive integer")
@icontract.ensure(lambda result: _finite_scalar(result), "updated EWA cost must be finite")
def nmf_minibatch_ewa_cost(
    previous_ewa_cost: float | None,
    batch_cost: float,
    batch_size: int,
    n_samples: int,
) -> float:
    """Update the exponentially weighted average cost used for minibatch stopping."""
    if previous_ewa_cost is None:
        return float(batch_cost)
    alpha = min(batch_size / (n_samples + 1), 1.0)
    return float(previous_ewa_cost * (1.0 - alpha) + float(batch_cost) * alpha)


@register_atom(witness_nmf_minibatch_h_change_converged)
@icontract.require(lambda H: _finite_matrix(H), "H must be a finite 2D matrix")
@icontract.require(lambda H, H_buffer: _same_shape(H_buffer, H), "H_buffer must be a finite 2D matrix with the same shape as H")
@icontract.require(lambda tol: _nonnegative_finite_scalar(tol), "tol must be finite and nonnegative")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def nmf_minibatch_h_change_converged(
    H: NDArray[np.float64],
    H_buffer: NDArray[np.float64],
    tol: float,
) -> bool:
    """Check sklearn's small-change stopping rule for successive H matrices."""
    current = np.asarray(H, dtype=np.float64)
    previous = np.asarray(H_buffer, dtype=np.float64)
    current_norm = float(np.linalg.norm(current))
    if current_norm == 0.0:
        return False
    h_diff = float(np.linalg.norm(current - previous) / current_norm)
    return bool(float(tol) > 0.0 and h_diff <= float(tol))


@register_atom(witness_nmf_minibatch_improvement_state)
@icontract.require(lambda ewa_cost: _finite_scalar(ewa_cost), "ewa_cost must be finite")
@icontract.require(lambda ewa_cost_min: ewa_cost_min is None or _finite_scalar(ewa_cost_min), "ewa_cost_min must be None or finite")
@icontract.require(lambda no_improvement: _nonnegative_int_or_none(no_improvement) and no_improvement is not None, "no_improvement must be a nonnegative integer")
@icontract.require(lambda max_no_improvement: _nonnegative_int_or_none(max_no_improvement), "max_no_improvement must be None or a nonnegative integer")
@icontract.ensure(
    lambda result: isinstance(result, tuple)
    and len(result) == 3
    and _finite_scalar(result[0])
    and _nonnegative_int_or_none(result[1])
    and isinstance(result[2], bool),
    "result must be (ewa_cost_min, no_improvement, should_stop) with finite and nonnegative bookkeeping values",
)
def nmf_minibatch_improvement_state(
    ewa_cost: float,
    ewa_cost_min: float | None,
    no_improvement: int,
    max_no_improvement: int | None,
) -> tuple[float, int, bool]:
    """Update sklearn's smoothed-cost improvement counters and stopping flag."""
    current = float(ewa_cost)
    if ewa_cost_min is None or current < float(ewa_cost_min):
        next_min = current
        next_count = 0
    else:
        next_min = float(ewa_cost_min)
        next_count = no_improvement + 1
    should_stop = max_no_improvement is not None and next_count >= max_no_improvement
    return next_min, next_count, should_stop
