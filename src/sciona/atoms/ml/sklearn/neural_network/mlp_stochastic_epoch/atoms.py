"""MLP stochastic-epoch helper atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_mlp_epoch_loss,
    witness_mlp_restore_best_parameters_required,
    witness_mlp_stochastic_incremental_break_required,
    witness_mlp_stochastic_max_iter_warning_required,
    witness_mlp_stochastic_no_improvement_count_after_trigger,
    witness_mlp_stochastic_stop_message,
    witness_mlp_time_step,
)


def _bool_valid(value: object) -> bool:
    return isinstance(value, bool)


def _finite_scalar(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value))


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


@register_atom(witness_mlp_epoch_loss)
@icontract.require(lambda accumulated_loss: _finite_scalar(accumulated_loss), "accumulated_loss must be finite")
@icontract.require(lambda n_training_samples: _positive_int(n_training_samples), "n_training_samples must be a positive integer")
@icontract.ensure(lambda result: _finite_scalar(result), "epoch loss must be finite")
def mlp_epoch_loss(
    accumulated_loss: float,
    *,
    n_training_samples: int,
) -> float:
    """Resolve sklearn's per-epoch stochastic loss from accumulated weighted minibatch loss."""
    return float(accumulated_loss) / float(n_training_samples)


@register_atom(witness_mlp_time_step)
@icontract.require(lambda current_time_step: _nonnegative_int(current_time_step), "current_time_step must be a nonnegative integer")
@icontract.require(lambda n_training_samples: _positive_int(n_training_samples), "n_training_samples must be a positive integer")
@icontract.ensure(lambda result: _positive_int(result), "updated time_step must be positive")
def mlp_time_step(
    current_time_step: int,
    *,
    n_training_samples: int,
) -> int:
    """Advance sklearn's stochastic MLP sample counter for one epoch."""
    return int(current_time_step) + int(n_training_samples)


@register_atom(witness_mlp_stochastic_stop_message)
@icontract.require(lambda early_stopping: _bool_valid(early_stopping), "early_stopping must be boolean")
@icontract.require(lambda tol: _finite_scalar(tol), "tol must be finite")
@icontract.require(lambda n_iter_no_change: _positive_int(n_iter_no_change), "n_iter_no_change must be a positive integer")
@icontract.ensure(lambda result: isinstance(result, str) and len(result) >= 1, "stop message must be nonempty")
def mlp_stochastic_stop_message(
    *,
    early_stopping: bool,
    tol: float,
    n_iter_no_change: int,
) -> str:
    """Format sklearn's stochastic MLP stopping message after too many unimproved epochs."""
    if early_stopping:
        return (
            "Validation score did not improve more than "
            "tol=%f for %d consecutive epochs."
            % (float(tol), int(n_iter_no_change))
        )
    return (
        "Training loss did not improve more than tol=%f"
        " for %d consecutive epochs."
        % (float(tol), int(n_iter_no_change))
    )


@register_atom(witness_mlp_stochastic_no_improvement_count_after_trigger)
@icontract.require(lambda is_stopping: _bool_valid(is_stopping), "is_stopping must be boolean")
@icontract.require(lambda no_improvement_count: _nonnegative_int(no_improvement_count), "no_improvement_count must be a nonnegative integer")
@icontract.ensure(lambda result: _nonnegative_int(result), "resulting no_improvement_count must be nonnegative")
def mlp_stochastic_no_improvement_count_after_trigger(
    *,
    is_stopping: bool,
    no_improvement_count: int,
) -> int:
    """Apply sklearn's no-improvement counter reset rule after optimizer trigger_stopping."""
    if is_stopping:
        return int(no_improvement_count)
    return 0


@register_atom(witness_mlp_stochastic_incremental_break_required)
@icontract.require(lambda incremental: _bool_valid(incremental), "incremental must be boolean")
@icontract.ensure(lambda result: _bool_valid(result), "incremental break flag must be boolean")
def mlp_stochastic_incremental_break_required(
    *,
    incremental: bool,
) -> bool:
    """Return whether sklearn exits the epoch loop immediately for partial_fit."""
    return bool(incremental)


@register_atom(witness_mlp_stochastic_max_iter_warning_required)
@icontract.require(lambda n_iter: _positive_int(n_iter), "n_iter must be a positive integer")
@icontract.require(lambda max_iter: _positive_int(max_iter), "max_iter must be a positive integer")
@icontract.require(lambda incremental: _bool_valid(incremental), "incremental must be boolean")
@icontract.ensure(lambda result: _bool_valid(result), "warning flag must be boolean")
def mlp_stochastic_max_iter_warning_required(
    n_iter: int,
    *,
    max_iter: int,
    incremental: bool,
) -> bool:
    """Return whether sklearn emits the stochastic max-iterations convergence warning."""
    return (not incremental) and int(n_iter) == int(max_iter)


@register_atom(witness_mlp_restore_best_parameters_required)
@icontract.require(lambda early_stopping: _bool_valid(early_stopping), "early_stopping must be boolean")
@icontract.ensure(lambda result: _bool_valid(result), "restore-best-parameters flag must be boolean")
def mlp_restore_best_parameters_required(
    *,
    early_stopping: bool,
) -> bool:
    """Return whether sklearn restores cached best parameters after stochastic training."""
    return bool(early_stopping)
