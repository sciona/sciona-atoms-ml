"""Spectral clustering public-wrapper atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_spectral_clustering_precomputed_affinity,
    witness_spectral_clustering_return_labels,
)


def _valid_parent_affinity(value: object) -> bool:
    return value is None or (isinstance(value, str) and value != "")


def _integer_vector(value: object) -> bool:
    array = np.asarray(value)
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.issubdtype(array.dtype, np.integer))


def _same_integer_vector(result: object, source: object) -> bool:
    lhs = np.asarray(result)
    rhs = np.asarray(source)
    return bool(lhs.ndim == 1 and lhs.shape == rhs.shape and np.array_equal(lhs, rhs) and np.issubdtype(lhs.dtype, np.integer))


@register_atom(witness_spectral_clustering_precomputed_affinity)
@icontract.require(
    lambda parent_affinity=None: _valid_parent_affinity(parent_affinity),
    "parent_affinity must be None or a nonempty string",
)
@icontract.ensure(
    lambda result: isinstance(result, str) and result == "precomputed",
    "wrapper affinity must be 'precomputed'",
)
def spectral_clustering_precomputed_affinity(
    parent_affinity: str | None = None,
) -> str:
    """Expose the fixed affinity passed by spectral_clustering into SpectralClustering."""
    del parent_affinity
    return "precomputed"


@register_atom(witness_spectral_clustering_return_labels)
@icontract.require(lambda labels: _integer_vector(labels), "labels must be a nonempty one-dimensional integer vector")
@icontract.ensure(lambda result, labels: _same_integer_vector(result, labels), "result must return the fitted label vector unchanged")
def spectral_clustering_return_labels(labels: NDArray[np.int_]) -> NDArray[np.int_]:
    """Return the fitted cluster labels exposed by spectral_clustering."""
    return np.asarray(labels).copy()
