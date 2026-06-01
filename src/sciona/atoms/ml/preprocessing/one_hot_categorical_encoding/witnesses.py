from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_build_category_registry(categorical_matrix: AbstractArray) -> AbstractArray:
    """Ghost witness for build_category_registry."""
    _ = (categorical_matrix)
    return AbstractArray(shape=categorical_matrix.shape, dtype=categorical_matrix.dtype)

def witness_encode_one_hot(categorical_matrix: AbstractArray, registries: AbstractArray, handle_unknown: AbstractScalar | str) -> AbstractArray:
    """Ghost witness for encode_one_hot."""
    _ = (categorical_matrix, registries, handle_unknown)
    return AbstractArray(shape=categorical_matrix.shape, dtype=categorical_matrix.dtype)

