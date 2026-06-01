from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_global_prior,
    witness_compute_oof_target_means,
)

@register_atom(witness_compute_global_prior, name="compute_global_prior")
@icontract.require(lambda targets: len(targets) > 0, "Precondition failed: len(targets) > 0")
@icontract.ensure(lambda result, targets: np.isfinite(prior), "Postcondition failed: np.isfinite(prior)")
def compute_global_prior(targets: NDArray[np.float64]) -> float:
    """Calculate global prior mean of the target variable.

    Args:
        targets: NDArray[np.float64]

    Returns:
        prior: float
    """
    import numpy
    return numpy.mean(targets=targets) # type: ignore

@register_atom(witness_compute_oof_target_means, name="compute_oof_target_means")
@icontract.require(lambda categorical_column, targets, prior, smoothing, train_indices, test_indices: len(categorical_column) == len(targets), "Precondition failed: len(categorical_column) == len(targets)")
@icontract.ensure(lambda result, categorical_column, targets, prior, smoothing, train_indices, test_indices: len(encoded_values) == len(test_indices), "Postcondition failed: len(encoded_values) == len(test_indices)")
def compute_oof_target_means(categorical_column: NDArray[Any], targets: NDArray[np.float64], prior: float, smoothing: float, train_indices: NDArray[np.int64], test_indices: NDArray[np.int64]) -> NDArray[np.float64]:
    """Calculate smoothed category averages using out-of-fold indexing.

    Args:
        categorical_column: NDArray[Any]
        targets: NDArray[np.float64]
        prior: float
        smoothing: float
        train_indices: NDArray[np.int64]
        test_indices: NDArray[np.int64]

    Returns:
        encoded_values: NDArray[np.float64]
    """
    import sklearn.preprocessing
    return sklearn.preprocessing.TargetEncoder(categorical_column=categorical_column, targets=targets, prior=prior, smoothing=smoothing, train_indices=train_indices, test_indices=test_indices) # type: ignore

