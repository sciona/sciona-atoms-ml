from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_generate_combination_indices,
    witness_compute_polynomial_products,
)

@register_atom(witness_generate_combination_indices, name="generate_combination_indices")
@icontract.require(lambda num_features, degree, interaction_only: num_features > 0, "Precondition failed: num_features > 0")
@icontract.require(lambda num_features, degree, interaction_only: degree >= 1, "Precondition failed: degree >= 1")
@icontract.ensure(lambda result, num_features, degree, interaction_only: len(index_combinations) > 0, "Postcondition failed: len(index_combinations) > 0")
def generate_combination_indices(num_features: int, degree: int, interaction_only: bool) -> int:
    """Calculate integer indices for polynomial combinations of a given size.

    Args:
        num_features: int
        degree: int
        interaction_only: bool

    Returns:
        index_combinations: list[tuple[int, ...]]
    """
    import math
    return math.comb(num_features=num_features, degree=degree, interaction_only=interaction_only) # type: ignore

@register_atom(witness_compute_polynomial_products, name="compute_polynomial_products")
@icontract.require(lambda matrix, index_combinations, include_bias: matrix.ndim == 2, "Precondition failed: matrix.ndim == 2")
@icontract.ensure(lambda result, matrix, index_combinations, include_bias: expanded_matrix.shape[0] == matrix.shape[0], "Postcondition failed: expanded_matrix.shape[0] == matrix.shape[0]")
def compute_polynomial_products(matrix: NDArray[np.float64], index_combinations: int, include_bias: bool) -> NDArray[np.float64]:
    """Multiply columns according to predefined combinations.

    Args:
        matrix: NDArray[np.float64]
        index_combinations: list[tuple[int, ...]]
        include_bias: bool

    Returns:
        expanded_matrix: NDArray[np.float64]
    """
    import numpy
    return numpy.prod(matrix=matrix, index_combinations=index_combinations, include_bias=include_bias) # type: ignore

