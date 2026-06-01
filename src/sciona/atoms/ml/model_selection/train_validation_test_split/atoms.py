from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_generate_split_indices,
    witness_slice_dataset_by_indices,
)

@register_atom(witness_generate_split_indices, name="generate_split_indices")
@icontract.require(lambda targets, train_ratio, val_ratio, test_ratio, stratify, groups: abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Precondition failed: abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6")
@icontract.ensure(lambda result, targets, train_ratio, val_ratio, test_ratio, stratify, groups: len(np.intersect1d(train_indices, val_indices)) == 0, "Postcondition failed: len(np.intersect1d(train_indices, val_indices)) == 0")
@icontract.ensure(lambda result, targets, train_ratio, val_ratio, test_ratio, stratify, groups: len(np.intersect1d(train_indices, test_indices)) == 0, "Postcondition failed: len(np.intersect1d(train_indices, test_indices)) == 0")
def generate_split_indices(targets: NDArray[Any], train_ratio: float, val_ratio: float, test_ratio: float, stratify: bool, groups: Optional[NDArray[Any]] = None) -> NDArray[np.int64]:
    """Generate disjoint training, validation, and test indices based on stratification or grouping constraints.

    Args:
        targets: 1D target array for stratification
        train_ratio: 0.0 < val < 1.0
        val_ratio: 0.0 <= val < 1.0
        test_ratio: 0.0 < val < 1.0
        stratify: bool
        groups: Optional[NDArray[Any]]

    Returns:
        train_indices: NDArray[np.int64]
    """
    import sklearn.model_selection
    return sklearn.model_selection.StratifiedShuffleSplit(targets=targets, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, stratify=stratify, groups=groups) # type: ignore

@register_atom(witness_slice_dataset_by_indices, name="slice_dataset_by_indices")
@icontract.require(lambda data, indices: data.ndim == 2, "Precondition failed: data.ndim == 2")
@icontract.require(lambda data, indices: np.max(indices) < data.shape[0], "Precondition failed: np.max(indices) < data.shape[0]")
@icontract.ensure(lambda result, data, indices: result.shape[0] == len(indices), "Postcondition failed: result.shape[0] == len(indices)")
def slice_dataset_by_indices(data: NDArray[np.float64], indices: NDArray[np.int64]) -> NDArray[np.float64]:
    """Slice a high-dimensional feature matrix using precalculated indices.

    Args:
        data: 2D array
        indices: NDArray[np.int64]

    Returns:
        result: NDArray[np.float64]
    """
    import numpy
    return numpy.take(data=data, indices=indices) # type: ignore

