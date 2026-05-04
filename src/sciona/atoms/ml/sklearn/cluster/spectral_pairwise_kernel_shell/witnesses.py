"""Ghost witnesses for SpectralClustering pairwise-kernel callback atoms."""

from __future__ import annotations

from collections.abc import Mapping

from sciona.ghost.abstract import AbstractArray


def witness_spectral_fit_pairwise_kernel_kwargs(
    affinity: object,
    params: Mapping[str, object],
) -> AbstractArray:
    """Describe the kwargs mapping passed into pairwise_kernels."""
    del affinity
    del params
    return AbstractArray(shape=(), dtype="object")


def witness_spectral_fit_pairwise_affinity_matrix(
    affinity_matrix: AbstractArray,
) -> AbstractArray:
    """Describe the dense affinity matrix returned by pairwise_kernels."""
    if len(affinity_matrix.shape) != 2:
        raise ValueError("affinity_matrix must be 2D")
    if affinity_matrix.shape[0] != affinity_matrix.shape[1]:
        raise ValueError("affinity_matrix must be square")
    return AbstractArray(
        shape=(int(affinity_matrix.shape[0]), int(affinity_matrix.shape[1])),
        dtype="float64",
    )
