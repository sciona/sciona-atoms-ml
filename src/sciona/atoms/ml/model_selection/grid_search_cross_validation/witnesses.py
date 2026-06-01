from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_generate_cartesian_grid(param_grid: AbstractScalar | str) -> AbstractScalar:
    """Ghost witness for generate_cartesian_grid."""
    _ = (param_grid)
    return AbstractScalar(dtype="float64")

def witness_fit_cv_candidate(estimator: AbstractScalar | Any, features: AbstractArray, targets: AbstractArray, params: AbstractScalar | str, train_indices: AbstractArray, val_indices: AbstractArray) -> AbstractScalar:
    """Ghost witness for fit_cv_candidate."""
    _ = (estimator, features, targets, params, train_indices, val_indices)
    return AbstractScalar(dtype="float64")

