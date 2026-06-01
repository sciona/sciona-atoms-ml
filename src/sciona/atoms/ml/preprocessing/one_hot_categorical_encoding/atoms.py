from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_build_category_registry,
    witness_encode_one_hot,
)

@register_atom(witness_build_category_registry, name="build_category_registry")
@icontract.require(lambda categorical_matrix: categorical_matrix.ndim == 2, "Precondition failed: categorical_matrix.ndim == 2")
@icontract.ensure(lambda result, categorical_matrix: len(registries) == categorical_matrix.shape[1], "Postcondition failed: len(registries) == categorical_matrix.shape[1]")
def build_category_registry(categorical_matrix: NDArray[Any]) -> list[NDArray[Any]]:
    """Extract sorted unique categories from a column matrix.

    Args:
        categorical_matrix: NDArray[Any]

    Returns:
        registries: list[NDArray[Any]]
    """
    import numpy
    return numpy.unique(categorical_matrix=categorical_matrix) # type: ignore

@register_atom(witness_encode_one_hot, name="encode_one_hot")
@icontract.require(lambda categorical_matrix, registries, handle_unknown: categorical_matrix.shape[1] == len(registries), "Precondition failed: categorical_matrix.shape[1] == len(registries)")
@icontract.ensure(lambda result, categorical_matrix, registries, handle_unknown: binary_matrix.ndim == 2, "Postcondition failed: binary_matrix.ndim == 2")
def encode_one_hot(categorical_matrix: NDArray[Any], registries: list[NDArray[Any]], handle_unknown: str) -> NDArray[np.float64]:
    """Project categories into sparse or dense binary dimensions using registry lists.

    Args:
        categorical_matrix: NDArray[Any]
        registries: list[NDArray[Any]]
        handle_unknown: str

    Returns:
        binary_matrix: NDArray[np.float64]
    """
    import sklearn.preprocessing
    return sklearn.preprocessing.OneHotEncoder(categorical_matrix=categorical_matrix, registries=registries, handle_unknown=handle_unknown) # type: ignore

