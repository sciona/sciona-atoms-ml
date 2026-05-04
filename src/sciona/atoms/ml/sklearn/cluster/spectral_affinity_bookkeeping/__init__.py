"""Deterministic SpectralClustering affinity bookkeeping helpers."""

from .atoms import (
    spectral_fit_pairwise_kernel_params,
    spectral_fit_symmetric_connectivity,
    spectral_fit_use_nearest_neighbors,
    spectral_fit_use_pairwise_kernel_hyperparameters,
    spectral_fit_use_precomputed_affinity,
    spectral_fit_use_precomputed_nearest_neighbors,
)

__all__ = [
    "spectral_fit_use_nearest_neighbors",
    "spectral_fit_use_precomputed_nearest_neighbors",
    "spectral_fit_use_precomputed_affinity",
    "spectral_fit_use_pairwise_kernel_hyperparameters",
    "spectral_fit_pairwise_kernel_params",
    "spectral_fit_symmetric_connectivity",
]
