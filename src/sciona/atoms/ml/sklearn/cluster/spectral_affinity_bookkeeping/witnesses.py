"""Ghost witnesses for SpectralClustering affinity-bookkeeping atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_spectral_fit_use_nearest_neighbors(affinity: str) -> AbstractArray:
    """Describe the nearest-neighbors affinity branch predicate."""
    del affinity
    return AbstractArray(shape=(), dtype="bool")


def witness_spectral_fit_use_precomputed_nearest_neighbors(affinity: str) -> AbstractArray:
    """Describe the precomputed-nearest-neighbors affinity branch predicate."""
    del affinity
    return AbstractArray(shape=(), dtype="bool")


def witness_spectral_fit_use_precomputed_affinity(affinity: str) -> AbstractArray:
    """Describe the precomputed-affinity branch predicate."""
    del affinity
    return AbstractArray(shape=(), dtype="bool")


def witness_spectral_fit_use_pairwise_kernel_hyperparameters(
    affinity: object,
) -> AbstractArray:
    """Describe whether SpectralClustering injects gamma, degree, and coef0."""
    del affinity
    return AbstractArray(shape=(), dtype="bool")


def witness_spectral_fit_pairwise_kernel_params(
    kernel_params: dict[str, object] | None,
    use_kernel_hyperparameters: bool,
    gamma: float,
    degree: float,
    coef0: float,
) -> AbstractArray:
    """Describe the resolved pairwise-kernel parameter mapping."""
    del kernel_params
    del use_kernel_hyperparameters
    del gamma
    del degree
    del coef0
    return AbstractArray(shape=(), dtype="object")


def witness_spectral_fit_symmetric_connectivity(connectivity: AbstractArray) -> AbstractArray:
    """Describe the symmetric connectivity matrix used for affinity_matrix_."""
    return AbstractArray(shape=connectivity.shape, dtype=connectivity.dtype)
