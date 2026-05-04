"""Ghost witnesses for spectral clustering fit-bookkeeping atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_spectral_fit_n_components(
    n_clusters: int,
    n_components: int | None = None,
) -> AbstractArray:
    """Describe SpectralClustering.fit's resolved embedding width."""
    del n_clusters
    del n_components
    return AbstractArray(shape=(), dtype="int64")


def witness_spectral_fit_verbose_message(assign_labels: str) -> AbstractArray:
    """Describe SpectralClustering.fit's verbose label-assignment message."""
    del assign_labels
    return AbstractArray(shape=(), dtype="object")


def witness_spectral_fit_use_kmeans(assign_labels: str) -> AbstractArray:
    """Describe whether SpectralClustering.fit uses k-means label assignment."""
    del assign_labels
    return AbstractArray(shape=(), dtype="bool")


def witness_spectral_fit_use_cluster_qr(assign_labels: str) -> AbstractArray:
    """Describe whether SpectralClustering.fit uses cluster_qr label assignment."""
    del assign_labels
    return AbstractArray(shape=(), dtype="bool")


def witness_spectral_fit_return_self(estimator_token: str) -> AbstractArray:
    """Describe SpectralClustering.fit returning self unchanged."""
    del estimator_token
    return AbstractArray(shape=(), dtype="object")
