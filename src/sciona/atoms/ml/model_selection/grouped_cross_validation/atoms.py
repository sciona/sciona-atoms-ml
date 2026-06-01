from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_partition_groups_to_folds,
)

@register_atom(witness_partition_groups_to_folds, name="partition_groups_to_folds")
@icontract.require(lambda groups, n_splits: len(np.unique(groups)) >= n_splits, "Precondition failed: len(np.unique(groups)) >= n_splits")
@icontract.ensure(lambda result, groups, n_splits: np.max(fold_assignments) < n_splits, "Postcondition failed: np.max(fold_assignments) < n_splits")
def partition_groups_to_folds(groups: NDArray[Any], n_splits: int) -> NDArray[np.int64]:
    """Distribute distinct groups across folds to minimize variance in fold sizes.

    Args:
        groups: NDArray[Any]
        n_splits: int

    Returns:
        fold_assignments: NDArray[np.int64]
    """
    import sklearn.model_selection.GroupKFold
    return sklearn.model_selection.GroupKFold._make_test_folds(groups=groups, n_splits=n_splits) # type: ignore

