"""Witnesses for t-SNE fit-shell helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_tsne_fit_return_self(estimator_token: str) -> AbstractArray:
    """Describe TSNE.fit returning self unchanged."""
    del estimator_token
    return AbstractArray(shape=(), dtype="object")
