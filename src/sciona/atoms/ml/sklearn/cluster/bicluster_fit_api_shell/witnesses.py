"""Ghost witnesses for spectral biclustering fit API-shell atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_bicluster_fit_accept_sparse_format(
    parent_accept_sparse: str | tuple[str, ...] | None = None,
) -> AbstractArray:
    """Describe the shared fit-time sparse format."""
    del parent_accept_sparse
    return AbstractArray(shape=(), dtype="object")


def witness_bicluster_fit_dtype_name(parent_dtype_name: str | None = None) -> AbstractArray:
    """Describe the shared fit-time dtype name."""
    del parent_dtype_name
    return AbstractArray(shape=(), dtype="object")


def witness_bicluster_sparse_input_tag(parent_sparse: bool) -> AbstractArray:
    """Describe the sparse-input tag override."""
    del parent_sparse
    return AbstractArray(shape=(), dtype="bool")
