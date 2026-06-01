from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_ksg_mutual_information,
)

@register_atom(witness_compute_ksg_mutual_information, name="compute_ksg_mutual_information")
@icontract.require(lambda features, targets, n_neighbors: features.shape[0] > n_neighbors, "Precondition failed: features.shape[0] > n_neighbors")
@icontract.ensure(lambda result, features, targets, n_neighbors: np.all(scores >= 0.0), "Postcondition failed: np.all(scores >= 0.0)")
def compute_ksg_mutual_information(features: NDArray[np.float64], targets: NDArray[Any], n_neighbors: int) -> NDArray[np.float64]:
    """Calculate Mutual Information scores using Kraskov nearest-neighbor entropy estimation.

    Args:
        features: NDArray[np.float64]
        targets: NDArray[Any]
        n_neighbors: int

    Returns:
        scores: NDArray[np.float64]
    """
    import sklearn.feature_selection._mutual_info
    return sklearn.feature_selection._mutual_info._estimate_mi(features=features, targets=targets, n_neighbors=n_neighbors) # type: ignore

