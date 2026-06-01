from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_stratified_kfold_indices,
)

@register_atom(witness_compute_stratified_kfold_indices, name="compute_stratified_kfold_indices")
@icontract.require(lambda targets, n_splits, shuffle, random_state: len(targets) >= n_splits, "Precondition failed: len(targets) >= n_splits")
@icontract.ensure(lambda result, targets, n_splits, shuffle, random_state: len(folds) == n_splits, "Postcondition failed: len(folds) == n_splits")
def compute_stratified_kfold_indices(targets: NDArray[Any], n_splits: int, shuffle: bool, random_state: int = None) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    """Generate list of train and test index pairs maintaining class ratios.

    Args:
        targets: NDArray[Any]
        n_splits: >= 2
        shuffle: bool
        random_state: Optional[int]

    Returns:
        folds: list[tuple[NDArray[np.int64], NDArray[np.int64]]]
    """
    import sklearn.model_selection
    return sklearn.model_selection.StratifiedKFold(targets=targets, n_splits=n_splits, shuffle=shuffle, random_state=random_state) # type: ignore

