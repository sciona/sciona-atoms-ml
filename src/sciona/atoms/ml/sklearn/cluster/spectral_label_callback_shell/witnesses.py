"""Ghost witnesses for SpectralClustering label-callback atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_spectral_fit_kmeans_kwargs(
    n_clusters: int,
    n_init: int,
    verbose: bool | int,
    random_state: AbstractArray,
) -> AbstractArray:
    """Describe the kwargs mapping passed into k_means."""
    del n_clusters
    del n_init
    del verbose
    del random_state
    return AbstractArray(shape=(), dtype="object")


def witness_spectral_fit_kmeans_output_labels(
    labels: AbstractArray,
) -> AbstractArray:
    """Describe the label vector unpacked from the k_means return tuple."""
    if len(labels.shape) != 1:
        raise ValueError("labels must be one-dimensional")
    return AbstractArray(shape=labels.shape, dtype="int64", min_val=0)


def witness_spectral_fit_discretize_kwargs(random_state: AbstractArray) -> AbstractArray:
    """Describe the kwargs mapping passed into discretize."""
    del random_state
    return AbstractArray(shape=(), dtype="object")
