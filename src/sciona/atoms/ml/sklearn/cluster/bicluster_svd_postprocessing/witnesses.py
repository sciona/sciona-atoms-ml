"""Ghost witnesses for deterministic biclustering SVD output shaping."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_bicluster_svd_left_vectors(
    u: AbstractArray,
    n_discard: int,
) -> AbstractArray:
    """Describe the kept left singular vectors returned by BaseSpectral._svd."""
    if len(u.shape) != 2:
        raise ValueError("u must be 2D")
    if n_discard < 0 or n_discard >= int(u.shape[1]):
        raise ValueError("n_discard must leave at least one left singular vector")
    return AbstractArray(
        shape=(int(u.shape[0]), int(u.shape[1]) - int(n_discard)),
        dtype="float64",
    )


def witness_bicluster_svd_right_vectors(
    vt: AbstractArray,
    n_discard: int,
) -> AbstractArray:
    """Describe the kept right singular vectors returned by BaseSpectral._svd."""
    if len(vt.shape) != 2:
        raise ValueError("vt must be 2D")
    if n_discard < 0 or n_discard >= int(vt.shape[0]):
        raise ValueError("n_discard must leave at least one right singular vector")
    return AbstractArray(
        shape=(int(vt.shape[1]), int(vt.shape[0]) - int(n_discard)),
        dtype="float64",
    )
