"""Ghost witnesses for spectral_clustering wrapper-constructor atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_spectral_clustering_constructor_kwargs(
    n_clusters: int,
    n_components: int | None,
    eigen_solver: str | None,
    random_state: object = None,
    n_init: int = 10,
    eigen_tol: float | str = "auto",
    assign_labels: str = "kmeans",
    verbose: bool | int = False,
) -> AbstractArray:
    """Describe the kwargs mapping passed into SpectralClustering by spectral_clustering."""
    del n_clusters
    del n_components
    del eigen_solver
    del random_state
    del n_init
    del eigen_tol
    del assign_labels
    del verbose
    return AbstractArray(shape=(), dtype="object")
