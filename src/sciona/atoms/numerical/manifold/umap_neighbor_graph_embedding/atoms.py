from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_build_fuzzy_simplicial_set,
    witness_optimize_umap_layout,
)

@register_atom(witness_build_fuzzy_simplicial_set, name="build_fuzzy_simplicial_set")
@icontract.require(lambda X, n_neighbors, metric: X.ndim == 2, "Precondition failed: X.ndim == 2")
@icontract.require(lambda X, n_neighbors, metric: n_neighbors > 1, "Precondition failed: n_neighbors > 1")
@icontract.ensure(lambda result, X, n_neighbors, metric: fuzzy_graph.shape == (X.shape[0], X.shape[0]), "Postcondition failed: fuzzy_graph.shape == (X.shape[0], X.shape[0])")
def build_fuzzy_simplicial_set(X: NDArray[np.float64], n_neighbors: int, metric: str = None) -> Any:
    """Construct high-dimensional symmetric fuzzy neighborhood matrices matching metric geometry.

    Args:
        X: NDArray[np.float64]
        n_neighbors: int
        metric: str

    Returns:
        fuzzy_graph: scipy.sparse.coo_matrix
    """
    import umap.umap_
    return umap.umap_.fuzzy_simplicial_set(X=X, n_neighbors=n_neighbors, metric=metric) # type: ignore

@register_atom(witness_optimize_umap_layout, name="optimize_umap_layout")
@icontract.require(lambda fuzzy_graph, n_epochs, min_dist: min_dist >= 0.0, "Precondition failed: min_dist >= 0.0")
@icontract.ensure(lambda result, fuzzy_graph, n_epochs, min_dist: embedding.shape[0] == fuzzy_graph.shape[0], "Postcondition failed: embedding.shape[0] == fuzzy_graph.shape[0]")
def optimize_umap_layout(fuzzy_graph: Any, n_epochs: int, min_dist: float) -> NDArray[np.float64]:
    """Optimize coordinates via stochastic gradient descent minimizing fuzzy set cross-entropy.

    Args:
        fuzzy_graph: scipy.sparse.coo_matrix
        n_epochs: int
        min_dist: float

    Returns:
        embedding: NDArray[np.float64]
    """
    import umap.umap_
    return umap.umap_.simplicial_set_embedding(fuzzy_graph=fuzzy_graph, n_epochs=n_epochs, min_dist=min_dist) # type: ignore

