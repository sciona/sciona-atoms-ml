"""t-SNE fit-value preparation helper atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_tsne_exact_distance_matrix,
    witness_tsne_exact_probability_vector,
    witness_tsne_provided_layout_matrix,
    witness_tsne_neighbor_graph_squared_data,
)


def _metric_name_valid(value: object) -> bool:
    return isinstance(value, str) and len(value) >= 1


def _finite_square_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 2
        and array.shape[0] == array.shape[1]
        and array.shape[0] >= 2
        and np.all(np.isfinite(array))
    )


def _finite_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _finite_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _same_shape(result: object, source: object) -> bool:
    return bool(np.asarray(result).shape == np.asarray(source).shape and np.all(np.isfinite(np.asarray(result, dtype=np.float64))))


@register_atom(witness_tsne_exact_distance_matrix)
@icontract.require(lambda distances: _finite_square_matrix(distances), "distances must be a finite square matrix")
@icontract.require(lambda metric: _metric_name_valid(metric), "metric must be a nonempty string")
@icontract.ensure(lambda result, distances: _same_shape(result, distances), "distance matrix must preserve the supplied square shape")
def tsne_exact_distance_matrix(
    distances: NDArray[np.float64],
    *,
    metric: str,
) -> NDArray[np.float64]:
    """Apply sklearn's exact-method distance postprocessing for a supplied distance matrix."""
    values = np.asarray(distances, dtype=np.float64).copy()
    if np.any(values < 0.0):
        raise ValueError("All distances should be positive, the metric given is not correct")
    if metric != "euclidean":
        values **= 2
    return np.asarray(values, dtype=np.float64)


@register_atom(witness_tsne_neighbor_graph_squared_data)
@icontract.require(lambda data: _finite_vector(data), "data must be a finite vector")
@icontract.require(lambda data: np.all(np.asarray(data, dtype=np.float64) >= 0.0), "data must be nonnegative")
@icontract.ensure(lambda result, data: _same_shape(result, data), "squared neighbor data must preserve the supplied vector shape")
def tsne_neighbor_graph_squared_data(
    data: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Square sklearn's Barnes-Hut neighbor-graph distance values in place semantics."""
    values = np.asarray(data, dtype=np.float64)
    return np.asarray(values**2, dtype=np.float64)


@register_atom(witness_tsne_exact_probability_vector)
@icontract.require(lambda P: _finite_vector(P), "P must be a finite vector")
@icontract.ensure(lambda result, P: _same_shape(result, P), "probability vector must preserve the supplied condensed shape")
def tsne_exact_probability_vector(
    P: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply sklearn's exact-method probability sanity checks to a supplied condensed probability vector."""
    values = np.asarray(P, dtype=np.float64)
    assert np.all(np.isfinite(values)), "All probabilities should be finite"
    assert np.all(values >= 0.0), "All probabilities should be non-negative"
    assert np.all(values <= 1.0), "All probabilities should be less or then equal to one"
    return np.asarray(values, dtype=np.float64)


@register_atom(witness_tsne_provided_layout_matrix)
@icontract.require(lambda init: _finite_matrix(init), "init must be a finite 2D matrix")
@icontract.ensure(lambda result, init: _same_shape(result, init), "init embedding must preserve the supplied matrix shape")
def tsne_provided_layout_matrix(
    init: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the provided starting matrix unchanged."""
    return np.asarray(init, dtype=np.float64)
