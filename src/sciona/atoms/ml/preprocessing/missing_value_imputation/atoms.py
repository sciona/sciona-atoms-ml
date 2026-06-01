from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_imputation_statistics,
    witness_fill_missing_values,
)

@register_atom(witness_compute_imputation_statistics, name="compute_imputation_statistics")
@icontract.require(lambda matrix, strategy: matrix.ndim == 2, "Precondition failed: matrix.ndim == 2")
@icontract.ensure(lambda result, matrix, strategy: len(statistics) == matrix.shape[1], "Postcondition failed: len(statistics) == matrix.shape[1]")
def compute_imputation_statistics(matrix: NDArray[np.float64], strategy: str) -> NDArray[np.float64]:
    """Identify non-nan values and calculate mean or median values per column.

    Args:
        matrix: NDArray[np.float64]
        strategy: str

    Returns:
        statistics: NDArray[np.float64]
    """
    import numpy
    return numpy.nanmean(matrix=matrix, strategy=strategy) # type: ignore

@register_atom(witness_fill_missing_values, name="fill_missing_values")
@icontract.require(lambda matrix, statistics: matrix.shape[1] == len(statistics), "Precondition failed: matrix.shape[1] == len(statistics)")
@icontract.ensure(lambda result, matrix, statistics: np.all(np.isfinite(imputed_matrix)), "Postcondition failed: np.all(np.isfinite(imputed_matrix))")
def fill_missing_values(matrix: NDArray[np.float64], statistics: NDArray[np.float64]) -> NDArray[np.float64]:
    """Replace NaN occurrences inside matrix with corresponding values from the statistics vector.

    Args:
        matrix: NDArray[np.float64]
        statistics: NDArray[np.float64]

    Returns:
        imputed_matrix: NDArray[np.float64]
    """
    import numpy
    return numpy.isnan(matrix=matrix, statistics=statistics) # type: ignore

