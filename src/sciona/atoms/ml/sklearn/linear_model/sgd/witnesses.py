"""Ghost witnesses for sklearn stochastic-gradient helper atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_vector(values: AbstractArray, name: str) -> int:
    if len(values.shape) != 1:
        raise ValueError(f"{name} must be 1D")
    size = int(values.shape[0])
    if size < 1:
        raise ValueError(f"{name} must be nonempty")
    return size


def _check_matrix(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    rows, cols = int(values.shape[0]), int(values.shape[1])
    if rows < 1 or cols < 2:
        raise ValueError(f"{name} must have samples and at least two classes")
    return rows, cols


def witness_sgd_l1_ratio_or_zero(l1_ratio: float | None) -> float:
    """Describe the scalar l1-ratio value passed to the SGD kernel."""
    del l1_ratio
    return 0.0


def witness_sgd_learning_rate_value(
    learning_rate: str,
    eta0: float,
    *,
    alpha: float = 0.0001,
    t: float = 1.0,
    power_t: float = 0.5,
    t0: float = 0.0,
) -> float:
    """Describe a scalar SGD learning-rate schedule value."""
    del learning_rate, eta0, alpha, t, power_t, t0
    return 0.0


def witness_sgd_passive_aggressive_step_size(
    loss_value: float,
    squared_norm: float,
    eta0: float,
    *,
    learning_rate: str,
) -> float:
    """Describe a passive-aggressive scalar step size."""
    del loss_value, squared_norm, eta0, learning_rate
    return 0.0


def witness_sgd_modified_huber_proba(scores: AbstractArray, *, binary: bool) -> AbstractArray:
    """Describe modified-Huber probability normalization from decision scores."""
    if binary:
        n_samples = _check_vector(scores, "scores")
        return AbstractArray(shape=(n_samples, 2), dtype="float64")
    n_samples, n_classes = _check_matrix(scores, "scores")
    return AbstractArray(shape=(n_samples, n_classes), dtype="float64")


def witness_passive_aggressive_classifier_sgd_config(loss: str) -> tuple[str, str, float]:
    """Describe classifier fit settings delegated to the SGD base class."""
    del loss
    return "hinge", "pa1", 1.0


def witness_passive_aggressive_regressor_sgd_config(loss: str) -> tuple[str, str, float]:
    """Describe regressor fit settings delegated to the SGD base class."""
    del loss
    return "epsilon_insensitive", "pa1", 1.0
