from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_class_imbalance_ratios(targets: AbstractArray) -> AbstractScalar:
    """Ghost witness for compute_class_imbalance_ratios."""
    _ = (targets)
    return AbstractScalar(dtype="float64")

def witness_generate_smote_samples(features: AbstractArray, targets: AbstractArray, k_neighbors: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for generate_smote_samples."""
    _ = (features, targets, k_neighbors)
    return AbstractArray(shape=features.shape, dtype=features.dtype)

