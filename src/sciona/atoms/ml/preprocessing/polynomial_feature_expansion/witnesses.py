from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_generate_combination_indices(num_features: AbstractScalar | int, degree: AbstractScalar | int, interaction_only: AbstractScalar | bool) -> AbstractScalar:
    """Ghost witness for generate_combination_indices."""
    _ = (num_features, degree, interaction_only)
    return AbstractScalar(dtype="float64")

def witness_compute_polynomial_products(matrix: AbstractArray, index_combinations: AbstractScalar | int, include_bias: AbstractScalar | bool) -> AbstractArray:
    """Ghost witness for compute_polynomial_products."""
    _ = (matrix, index_combinations, include_bias)
    return AbstractArray(shape=matrix.shape, dtype=matrix.dtype)

