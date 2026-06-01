from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_fit_transformer_stage(data: AbstractArray, transformer_template: AbstractScalar | Any, targets: AbstractArray) -> AbstractScalar:
    """Ghost witness for fit_transformer_stage."""
    _ = (data, transformer_template, targets)
    return AbstractScalar(dtype="float64")

def witness_transform_only_stage(data: AbstractArray, fitted_state: AbstractScalar | Any) -> AbstractArray:
    """Ghost witness for transform_only_stage."""
    _ = (data, fitted_state)
    return AbstractArray(shape=data.shape, dtype=data.dtype)

