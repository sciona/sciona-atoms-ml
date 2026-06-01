from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_raw_residuals,
    witness_compute_l2_regression_metrics,
)

@register_atom(witness_compute_raw_residuals, name="compute_raw_residuals")
@icontract.require(lambda y_true, y_pred: y_true.shape == y_pred.shape, "Precondition failed: y_true.shape == y_pred.shape")
@icontract.ensure(lambda result, y_true, y_pred: residuals.shape == y_true.shape, "Postcondition failed: residuals.shape == y_true.shape")
def compute_raw_residuals(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculate element-wise residual errors.

    Args:
        y_true: NDArray[np.float64]
        y_pred: NDArray[np.float64]

    Returns:
        residuals: NDArray[np.float64]
    """
    import numpy
    return numpy.subtract(y_true=y_true, y_pred=y_pred) # type: ignore

@register_atom(witness_compute_l2_regression_metrics, name="compute_l2_regression_metrics")
@icontract.require(lambda y_true, residuals: y_true is not None, "Precondition failed: y_true is not None")
@icontract.ensure(lambda result, y_true, residuals: result is not None, "Postcondition failed: result is not None")
def compute_l2_regression_metrics(y_true: NDArray[np.float64], residuals: NDArray[np.float64]) -> float:
    """Compute MSE, RMSE, and R2 coefficients from residuals.

    Args:
        y_true: NDArray[np.float64]
        residuals: NDArray[np.float64]

    Returns:
        mse: float
    """
    import sklearn.metrics
    return sklearn.metrics.mean_squared_error(y_true=y_true, residuals=residuals) # type: ignore

