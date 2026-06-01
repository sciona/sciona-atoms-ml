from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_build_hnsw_index,
    witness_query_hnsw_index,
)

@register_atom(witness_build_hnsw_index, name="build_hnsw_index")
@icontract.require(lambda data, M, ef_construction: data.ndim == 2, "Precondition failed: data.ndim == 2")
@icontract.ensure(lambda result, data, M, ef_construction: index is not None, "Postcondition failed: index is not None")
def build_hnsw_index(data: NDArray[np.float64], M: int = None, ef_construction: int = None) -> Any:
    """Build a multi-layer hierarchical navigable small world graph index.

    Args:
        data: NDArray[np.float64]
        M: int
        ef_construction: int

    Returns:
        index: Any
    """
    import hnswlib
    return hnswlib.Index(data=data, M=M, ef_construction=ef_construction) # type: ignore

@register_atom(witness_query_hnsw_index, name="query_hnsw_index")
@icontract.require(lambda index, query_points, k, ef_search: k >= 1, "Precondition failed: k >= 1")
@icontract.ensure(lambda result, index, query_points, k, ef_search: distances.shape == indices.shape, "Postcondition failed: distances.shape == indices.shape")
def query_hnsw_index(index: Any, query_points: NDArray[np.float64], k: int = None, ef_search: int = None) -> NDArray[np.float64]:
    """Query the built HNSW graph for fast approximate nearest neighbors.

    Args:
        index: Any
        query_points: NDArray[np.float64]
        k: int
        ef_search: int

    Returns:
        distances: NDArray[np.float64]
    """
    import hnswlib.Index
    return hnswlib.Index.knn_query(index=index, query_points=query_points, k=k, ef_search=ef_search) # type: ignore

