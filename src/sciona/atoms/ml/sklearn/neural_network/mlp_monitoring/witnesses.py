"""Ghost witnesses for MLP stochastic-monitor helper atoms."""

from __future__ import annotations


def witness_mlp_monitor_defaults(
    *,
    early_stopping: bool,
) -> tuple[float | None, float | None, int, tuple[float, ...] | None]:
    """Describe stochastic-monitor defaults after MLP first-pass initialization."""
    del early_stopping
    return None, None, 0, None


def witness_mlp_monitor_best_loss(
    last_loss: float,
    best_loss: float,
) -> float:
    """Describe the best-loss update in the non-early-stopping branch."""
    del last_loss
    return best_loss


def witness_mlp_monitor_loss_no_improvement_count(
    last_loss: float,
    best_loss: float,
    *,
    tol: float,
    no_improvement_count: int,
) -> int:
    """Describe the loss-based no-improvement counter update."""
    del last_loss
    del best_loss
    del tol
    return no_improvement_count


def witness_mlp_monitor_best_validation_score(
    last_validation_score: float,
    best_validation_score: float,
) -> float:
    """Describe the best-validation-score update in the early-stopping branch."""
    del last_validation_score
    return best_validation_score


def witness_mlp_monitor_validation_no_improvement_count(
    last_validation_score: float,
    best_validation_score: float,
    *,
    tol: float,
    no_improvement_count: int,
) -> int:
    """Describe the validation-score no-improvement counter update."""
    del last_validation_score
    del best_validation_score
    del tol
    return no_improvement_count
