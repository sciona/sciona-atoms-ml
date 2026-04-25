"""MLP stochastic-monitor helper atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_mlp_monitor_best_loss,
    witness_mlp_monitor_best_validation_score,
    witness_mlp_monitor_defaults,
    witness_mlp_monitor_loss_no_improvement_count,
    witness_mlp_monitor_validation_no_improvement_count,
)

MonitorDefaults = tuple[float | None, float | None, int, tuple[float, ...] | None]


def _bool_valid(value: object) -> bool:
    return isinstance(value, bool)


def _finite_scalar(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value))


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _defaults_valid(result: MonitorDefaults) -> bool:
    if not isinstance(result, tuple) or len(result) != 4:
        return False
    best_loss, best_validation_score, no_improvement_count, validation_scores = result
    if not _nonnegative_int(no_improvement_count) or no_improvement_count != 0:
        return False
    if validation_scores is not None and validation_scores != ():
        return False
    if best_loss is None:
        return bool(best_validation_score == -np.inf and validation_scores == ())
    return bool(np.isinf(best_loss) and best_loss > 0 and best_validation_score is None and validation_scores is None)


@register_atom(witness_mlp_monitor_defaults)
@icontract.require(lambda early_stopping: _bool_valid(early_stopping), "early_stopping must be boolean")
@icontract.ensure(lambda result: _defaults_valid(result), "monitor defaults must match sklearn's stochastic initialization branch")
def mlp_monitor_defaults(
    *,
    early_stopping: bool,
) -> MonitorDefaults:
    """Return sklearn's stochastic MLP monitor defaults after first-pass initialization."""
    if early_stopping:
        return None, float(-np.inf), 0, ()
    return float(np.inf), None, 0, None


@register_atom(witness_mlp_monitor_best_loss)
@icontract.require(lambda last_loss: _finite_scalar(last_loss), "last_loss must be finite")
@icontract.require(lambda best_loss: _finite_scalar(best_loss), "best_loss must be finite")
@icontract.ensure(lambda result: _finite_scalar(result), "best_loss update must stay finite")
def mlp_monitor_best_loss(
    last_loss: float,
    best_loss: float,
) -> float:
    """Update the tracked best training loss in the non-early-stopping branch."""
    return float(last_loss) if float(last_loss) < float(best_loss) else float(best_loss)


@register_atom(witness_mlp_monitor_loss_no_improvement_count)
@icontract.require(lambda last_loss: _finite_scalar(last_loss), "last_loss must be finite")
@icontract.require(lambda best_loss: _finite_scalar(best_loss), "best_loss must be finite")
@icontract.require(lambda tol: _finite_scalar(tol) and float(tol) >= 0.0, "tol must be finite and nonnegative")
@icontract.require(lambda no_improvement_count: _nonnegative_int(no_improvement_count), "no_improvement_count must be a nonnegative integer")
@icontract.ensure(lambda result: _nonnegative_int(result), "updated no_improvement_count must be nonnegative")
def mlp_monitor_loss_no_improvement_count(
    last_loss: float,
    best_loss: float,
    *,
    tol: float,
    no_improvement_count: int,
) -> int:
    """Update sklearn's no-improvement counter from the latest training loss."""
    if float(last_loss) > float(best_loss) - float(tol):
        return int(no_improvement_count) + 1
    return 0


@register_atom(witness_mlp_monitor_best_validation_score)
@icontract.require(lambda last_validation_score: _finite_scalar(last_validation_score), "last_validation_score must be finite")
@icontract.require(lambda best_validation_score: _finite_scalar(best_validation_score) or best_validation_score == -np.inf, "best_validation_score must be finite or -inf")
@icontract.ensure(lambda result: _finite_scalar(result) or result == -np.inf, "best validation score must remain finite or -inf")
def mlp_monitor_best_validation_score(
    last_validation_score: float,
    best_validation_score: float,
) -> float:
    """Update the tracked best validation score in the early-stopping branch."""
    return float(last_validation_score) if float(last_validation_score) > float(best_validation_score) else float(best_validation_score)


@register_atom(witness_mlp_monitor_validation_no_improvement_count)
@icontract.require(lambda last_validation_score: _finite_scalar(last_validation_score), "last_validation_score must be finite")
@icontract.require(lambda best_validation_score: _finite_scalar(best_validation_score) or best_validation_score == -np.inf, "best_validation_score must be finite or -inf")
@icontract.require(lambda tol: _finite_scalar(tol) and float(tol) >= 0.0, "tol must be finite and nonnegative")
@icontract.require(lambda no_improvement_count: _nonnegative_int(no_improvement_count), "no_improvement_count must be a nonnegative integer")
@icontract.ensure(lambda result: _nonnegative_int(result), "updated no_improvement_count must be nonnegative")
def mlp_monitor_validation_no_improvement_count(
    last_validation_score: float,
    best_validation_score: float,
    *,
    tol: float,
    no_improvement_count: int,
) -> int:
    """Update sklearn's no-improvement counter from the latest validation score."""
    if float(last_validation_score) < float(best_validation_score) + float(tol):
        return int(no_improvement_count) + 1
    return 0
