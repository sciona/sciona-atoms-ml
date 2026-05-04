"""Spectral clustering pairwise-kernel callback-shell atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Mapping

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_spectral_fit_pairwise_affinity_matrix,
    witness_spectral_fit_pairwise_kernel_kwargs,
)


def _nonempty_string(value: object) -> bool:
    return bool(isinstance(value, str) and value != "")


def _callable_or_nonempty_string(value: object) -> bool:
    return callable(value) or _nonempty_string(value)


def _kernel_params_mapping(value: object) -> bool:
    return bool(
        isinstance(value, Mapping)
        and all(isinstance(key, str) and key != "" for key in value)
    )


def _pairwise_kernel_kwargs_valid(
    result: object,
    affinity: object,
    params: Mapping[str, object],
) -> bool:
    return bool(
        isinstance(result, dict)
        and set(result) == {"metric", "filter_params", *params.keys()}
        and result["metric"] is affinity
        and result["filter_params"] is True
        and all(result[key] == value for key, value in params.items())
    )


def _finite_square_matrix(value: object) -> bool:
    array = np.asarray(value, dtype=np.float64)
    return bool(
        array.ndim == 2
        and array.shape[0] >= 1
        and array.shape[0] == array.shape[1]
        and np.all(np.isfinite(array))
    )


def _same_shape_and_values(result: object, value: object) -> bool:
    lhs = np.asarray(result, dtype=np.float64)
    rhs = np.asarray(value, dtype=np.float64)
    return bool(lhs.shape == rhs.shape and np.array_equal(lhs, rhs))


@register_atom(witness_spectral_fit_pairwise_kernel_kwargs)
@icontract.require(
    lambda affinity: _callable_or_nonempty_string(affinity),
    "affinity must be a callable or a nonempty string",
)
@icontract.require(
    lambda params: _kernel_params_mapping(params),
    "params must be a mapping with nonempty string keys",
)
@icontract.ensure(
    lambda result, affinity, params: _pairwise_kernel_kwargs_valid(result, affinity, params),
    "result must match SpectralClustering's pairwise_kernels kwargs",
)
def spectral_fit_pairwise_kernel_kwargs(
    affinity: object,
    params: Mapping[str, object],
) -> dict[str, object]:
    """Resolve the pairwise_kernels kwargs used by SpectralClustering.fit."""
    return {
        "metric": affinity,
        "filter_params": True,
        **dict(params),
    }


@register_atom(witness_spectral_fit_pairwise_affinity_matrix)
@icontract.require(
    lambda affinity_matrix: _finite_square_matrix(affinity_matrix),
    "affinity_matrix must be a nonempty finite square dense matrix",
)
@icontract.ensure(
    lambda result, affinity_matrix: _same_shape_and_values(result, affinity_matrix),
    "result must preserve the pairwise-kernel affinity matrix",
)
def spectral_fit_pairwise_affinity_matrix(
    affinity_matrix: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Expose the dense affinity matrix returned by the deferred pairwise_kernels call."""
    return np.asarray(affinity_matrix, dtype=np.float64).copy()
