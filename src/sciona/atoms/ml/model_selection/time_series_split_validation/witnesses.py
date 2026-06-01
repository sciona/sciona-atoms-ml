from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_time_series_split_indices(n_samples: AbstractScalar | int, n_splits: AbstractScalar | int, gap: AbstractScalar | int, max_train_size: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for compute_time_series_split_indices."""
    _ = (n_samples, n_splits, gap, max_train_size)
    return AbstractArray(shape=(), dtype="float64")

