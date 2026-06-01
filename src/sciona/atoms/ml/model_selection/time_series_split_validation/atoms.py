from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_time_series_split_indices,
)

@register_atom(witness_compute_time_series_split_indices, name="compute_time_series_split_indices")
@icontract.require(lambda n_samples, n_splits, gap, max_train_size: n_samples > n_splits + gap, "Precondition failed: n_samples > n_splits + gap")
@icontract.ensure(lambda result, n_samples, n_splits, gap, max_train_size: len(folds) == n_splits, "Postcondition failed: len(folds) == n_splits")
def compute_time_series_split_indices(n_samples: int, n_splits: int, gap: int, max_train_size: int = None) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    """Generate train/test boundary indexes for time-series expanding or rolling window validation.

    Args:
        n_samples: int
        n_splits: int
        gap: int
        max_train_size: Optional[int]

    Returns:
        folds: list[tuple[NDArray[np.int64], NDArray[np.int64]]]
    """
    import sklearn.model_selection
    return sklearn.model_selection.TimeSeriesSplit(n_samples=n_samples, n_splits=n_splits, gap=gap, max_train_size=max_train_size) # type: ignore

