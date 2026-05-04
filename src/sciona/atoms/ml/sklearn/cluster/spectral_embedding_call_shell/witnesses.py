"""Ghost witnesses for SpectralClustering embedding-call atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_spectral_fit_embedding_random_state(random_state: int | None = None) -> AbstractArray:
    """Describe the normalized RandomState passed into _spectral_embedding."""
    del random_state
    return AbstractArray(shape=(), dtype="object")


def witness_spectral_fit_embedding_drop_first(parent_drop_first: bool = True) -> AbstractArray:
    """Describe the fixed drop_first flag used by SpectralClustering.fit."""
    del parent_drop_first
    return AbstractArray(shape=(), dtype="bool")


def witness_spectral_fit_embedding_call_kwargs(
    n_components: int,
    eigen_solver: str | None,
    random_state: int | None = None,
    eigen_tol: float | str = "auto",
) -> AbstractArray:
    """Describe the kwargs mapping passed into _spectral_embedding."""
    del n_components
    del eigen_solver
    del random_state
    del eigen_tol
    return AbstractArray(shape=(), dtype="object")
