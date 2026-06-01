from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_ksg_mutual_information(features: AbstractArray, targets: AbstractArray, n_neighbors: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for compute_ksg_mutual_information."""
    _ = (features, targets, n_neighbors)
    return AbstractArray(shape=features.shape, dtype=features.dtype)

