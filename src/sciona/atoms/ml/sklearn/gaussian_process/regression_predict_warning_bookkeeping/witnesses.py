"""Ghost witnesses for Gaussian-process regression predict warning helpers."""

from __future__ import annotations

from numpy.typing import NDArray


def witness_gp_predict_negative_variance_mask(y_var: NDArray[float]) -> NDArray[bool]:
    """Describe the negative-variance mask before predictive clipping."""
    del y_var
    raise NotImplementedError


def witness_gp_predict_negative_variance_warning_required(
    negative_mask: NDArray[bool],
) -> bool:
    """Describe whether negative predictive variances trigger a warning."""
    del negative_mask
    return False


def witness_gp_predict_nonnegative_variance(
    y_var: NDArray[float],
    negative_mask: NDArray[bool],
) -> NDArray[float]:
    """Describe the clipped predictive-variance vector."""
    del y_var
    del negative_mask
    raise NotImplementedError
