"""Ghost witnesses for selected sklearn calibration atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_calibration_curve(
    y_true: AbstractArray,
    y_prob: AbstractArray,
    *,
    pos_label: int | float | bool | str | None = None,
    n_bins: int = 5,
    strategy: str = "uniform",
) -> tuple[AbstractArray, AbstractArray]:
    """Describe true and predicted probability vectors from calibration bins."""
    del pos_label
    if len(y_true.shape) != 1:
        raise ValueError("y_true must be 1D")
    if len(y_prob.shape) != 1:
        raise ValueError("y_prob must be 1D")
    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError("y_true and y_prob must have equal sample count")
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    if strategy not in {"uniform", "quantile"}:
        raise ValueError("strategy must be 'uniform' or 'quantile'")
    prob_true = AbstractArray(shape=(n_bins,), dtype="float64", min_val=0.0, max_val=1.0)
    prob_pred = AbstractArray(shape=(n_bins,), dtype="float64", min_val=0.0, max_val=1.0)
    return prob_true, prob_pred
