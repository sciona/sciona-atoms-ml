"""Ghost witnesses for spectral clustering fit API-shell atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_spectral_fit_accept_sparse_formats(
    parent_accept_sparse: tuple[str, ...] | None = None,
) -> AbstractArray:
    """Describe the accepted sparse formats for SpectralClustering.fit validation."""
    del parent_accept_sparse
    return AbstractArray(shape=(3,), dtype="object")


def witness_spectral_fit_dtype_name(parent_dtype_name: str | None = None) -> AbstractArray:
    """Describe the fit-validation dtype name for SpectralClustering."""
    del parent_dtype_name
    return AbstractArray(shape=(), dtype="object")


def witness_spectral_fit_affinity_allows_square_input(affinity: str) -> AbstractArray:
    """Describe whether the affinity allows square inputs without warning."""
    del affinity
    return AbstractArray(shape=(), dtype="bool")


def witness_spectral_fit_square_input_warning_required(
    affinity: str,
    shape: tuple[int, int],
) -> AbstractArray:
    """Describe the square-input warning predicate for SpectralClustering.fit."""
    del affinity
    del shape
    return AbstractArray(shape=(), dtype="bool")


def witness_spectral_pairwise_input_tag(affinity: str, parent_pairwise: bool) -> AbstractArray:
    """Describe the pairwise-input tag override for SpectralClustering."""
    del affinity
    del parent_pairwise
    return AbstractArray(shape=(), dtype="bool")
