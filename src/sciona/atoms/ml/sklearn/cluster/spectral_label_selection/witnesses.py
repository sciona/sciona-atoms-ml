"""Ghost witnesses for SpectralClustering label-selection atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_spectral_fit_use_discretize(assign_labels: str) -> AbstractArray:
    """Describe the fallback discretize branch predicate."""
    del assign_labels
    return AbstractArray(shape=(), dtype="bool")


def witness_spectral_fit_selected_labels(
    use_kmeans: bool,
    use_cluster_qr: bool,
    kmeans_labels: AbstractArray,
    cluster_qr_labels: AbstractArray,
    discretize_labels: AbstractArray,
) -> AbstractArray:
    """Describe the final selected SpectralClustering label vector."""
    del use_kmeans
    del use_cluster_qr
    if len(kmeans_labels.shape) != 1 or len(cluster_qr_labels.shape) != 1 or len(discretize_labels.shape) != 1:
        raise ValueError("all label vectors must be one-dimensional")
    n_samples = int(kmeans_labels.shape[0])
    if int(cluster_qr_labels.shape[0]) != n_samples or int(discretize_labels.shape[0]) != n_samples:
        raise ValueError("all label vectors must share the same length")
    return AbstractArray(shape=(n_samples,), dtype="int64", min_val=0)
