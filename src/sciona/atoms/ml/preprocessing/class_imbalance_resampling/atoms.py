from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_class_imbalance_ratios,
    witness_generate_smote_samples,
)

@register_atom(witness_compute_class_imbalance_ratios, name="compute_class_imbalance_ratios")
@icontract.require(lambda targets: targets is not None, "Precondition failed: targets is not None")
@icontract.ensure(lambda result, targets: result is not None, "Postcondition failed: result is not None")
def compute_class_imbalance_ratios(targets: NDArray[np.int64]) -> bool:
    """Verify class frequencies and determine resampling target counts.

    Args:
        targets: NDArray[np.int64]

    Returns:
        imbalance_detected: bool
    """
    import numpy
    return numpy.bincount(targets=targets) # type: ignore

@register_atom(witness_generate_smote_samples, name="generate_smote_samples")
@icontract.require(lambda features, targets, k_neighbors: features.shape[0] == len(targets), "Precondition failed: features.shape[0] == len(targets)")
@icontract.ensure(lambda result, features, targets, k_neighbors: resampled_features.shape[0] > features.shape[0], "Postcondition failed: resampled_features.shape[0] > features.shape[0]")
def generate_smote_samples(features: NDArray[np.float64], targets: NDArray[np.int64], k_neighbors: int) -> NDArray[np.float64]:
    """Create synthetic minority samples using linear interpolation of nearest neighbors.

    Args:
        features: NDArray[np.float64]
        targets: NDArray[np.int64]
        k_neighbors: int

    Returns:
        resampled_features: NDArray[np.float64]
    """
    import imblearn.over_sampling
    return imblearn.over_sampling.SMOTE(features=features, targets=targets, k_neighbors=k_neighbors) # type: ignore

