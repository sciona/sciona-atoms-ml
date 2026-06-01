from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_build_hnsw_index(data: AbstractArray, M: AbstractScalar | int, ef_construction: AbstractScalar | int) -> AbstractScalar:
    """Ghost witness for build_hnsw_index."""
    _ = (data, M, ef_construction)
    return AbstractScalar(dtype="float64")

def witness_query_hnsw_index(index: AbstractScalar | Any, query_points: AbstractArray, k: AbstractScalar | int, ef_search: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for query_hnsw_index."""
    _ = (index, query_points, k, ef_search)
    return AbstractArray(shape=query_points.shape, dtype=query_points.dtype)

