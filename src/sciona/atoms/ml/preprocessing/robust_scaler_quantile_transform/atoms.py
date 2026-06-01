from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_column_percentiles,
    witness_apply_robust_normalization,
)

@register_atom(witness_compute_column_percentiles, name="compute_column_percentiles")
@icontract.require(lambda matrix, percentiles: matrix.ndim == 2, "Precondition failed: matrix.ndim == 2")
@icontract.ensure(lambda result, matrix, percentiles: calculated_quantiles.shape == (len(percentiles), matrix.shape[1]), "Postcondition failed: calculated_quantiles.shape == (len(percentiles), matrix.shape[1])")
def compute_column_percentiles(matrix: NDArray[np.float64], percentiles: NDArray[np.float64]) -> NDArray[np.float64]:
    """Extract target quantiles from each column of a matrix.

    Args:
        matrix: NDArray[np.float64]
        percentiles: values in [0.0, 100.0]

    Returns:
        calculated_quantiles: NDArray[np.float64]
    """
    import numpy
    return numpy.percentile(matrix=matrix, percentiles=percentiles) # type: ignore

@register_atom(witness_apply_robust_normalization, name="apply_robust_normalization")
@icontract.require(lambda matrix, medians, iqrs: matrix.shape[1] == len(medians), "Precondition failed: matrix.shape[1] == len(medians)")
@icontract.ensure(lambda result, matrix, medians, iqrs: result.shape == matrix.shape, "Postcondition failed: result.shape == matrix.shape")
def apply_robust_normalization(matrix: NDArray[np.float64], medians: NDArray[np.float64], iqrs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Shift matrix by median and scale by IQR.

    Args:
        matrix: NDArray[np.float64]
        medians: NDArray[np.float64]
        iqrs: NDArray[np.float64]

    Returns:
        result: NDArray[np.float64]
    """
    import sklearn.preprocessing
    return sklearn.preprocessing.RobustScaler(matrix=matrix, medians=medians, iqrs=iqrs) # type: ignore

