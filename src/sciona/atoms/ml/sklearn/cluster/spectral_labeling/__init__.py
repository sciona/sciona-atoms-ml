"""Estimator-independent sklearn spectral-clustering label assignment atoms."""

from .atoms import (
    spectral_cluster_qr_labels,
    spectral_discretize_labels,
)

__all__ = [
    "spectral_cluster_qr_labels",
    "spectral_discretize_labels",
]
