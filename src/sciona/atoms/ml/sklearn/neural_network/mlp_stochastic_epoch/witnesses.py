"""Ghost witnesses for MLP stochastic-epoch helper atoms."""

from __future__ import annotations


def witness_mlp_epoch_loss(
    accumulated_loss: float,
    *,
    n_training_samples: int,
) -> float:
    """Describe the resolved stochastic epoch loss."""
    del accumulated_loss
    del n_training_samples
    return 0.0


def witness_mlp_time_step(
    current_time_step: int,
    *,
    n_training_samples: int,
) -> int:
    """Describe the updated stochastic sample counter."""
    del n_training_samples
    return current_time_step


def witness_mlp_stochastic_stop_message(
    *,
    early_stopping: bool,
    tol: float,
    n_iter_no_change: int,
) -> str:
    """Describe the stopping message after too many unimproved epochs."""
    del early_stopping
    del tol
    del n_iter_no_change
    return "message"


def witness_mlp_stochastic_no_improvement_count_after_trigger(
    *,
    is_stopping: bool,
    no_improvement_count: int,
) -> int:
    """Describe the no-improvement counter after optimizer stopping logic runs."""
    del is_stopping
    return no_improvement_count


def witness_mlp_stochastic_incremental_break_required(
    *,
    incremental: bool,
) -> bool:
    """Describe whether stochastic MLP training breaks after one epoch."""
    return incremental


def witness_mlp_stochastic_max_iter_warning_required(
    n_iter: int,
    *,
    max_iter: int,
    incremental: bool,
) -> bool:
    """Describe whether sklearn emits the stochastic max-iteration warning."""
    del n_iter
    del max_iter
    return incremental


def witness_mlp_restore_best_parameters_required(
    *,
    early_stopping: bool,
) -> bool:
    """Describe whether cached best parameters are restored after training."""
    return early_stopping
