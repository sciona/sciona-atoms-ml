from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_generate_split_indices(targets: AbstractArray, train_ratio: AbstractScalar | float, val_ratio: AbstractScalar | float, test_ratio: AbstractScalar | float, stratify: AbstractScalar | bool, groups: AbstractArray) -> AbstractArray:
    """Ghost witness for generate_split_indices."""
    _ = (targets, train_ratio, val_ratio, test_ratio, stratify, groups)
    return AbstractArray(shape=targets.shape, dtype=targets.dtype)

def witness_slice_dataset_by_indices(data: AbstractArray, indices: AbstractArray) -> AbstractArray:
    """Ghost witness for slice_dataset_by_indices."""
    _ = (data, indices)
    return AbstractArray(shape=data.shape, dtype=data.dtype)

