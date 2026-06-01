from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_generate_bootstrap_indices(n_samples: AbstractScalar | int, random_state: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for generate_bootstrap_indices."""
    _ = (n_samples, random_state)
    return AbstractArray(shape=(), dtype="float64")

def witness_fit_bootstrap_estimator(estimator_template: AbstractScalar | Any, features: AbstractArray, targets: AbstractArray, bootstrap_indices: AbstractArray) -> AbstractScalar:
    """Ghost witness for fit_bootstrap_estimator."""
    _ = (estimator_template, features, targets, bootstrap_indices)
    return AbstractScalar(dtype="float64")

