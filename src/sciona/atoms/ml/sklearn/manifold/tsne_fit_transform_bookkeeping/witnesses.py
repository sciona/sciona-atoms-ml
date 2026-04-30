"""Ghost witnesses for sklearn t-SNE fit-transform bookkeeping helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_tsne_fit_transform_require_single_iter_source(
    n_iter: str | int,
    max_iter: int | None,
) -> bool:
    """Describe the boolean success result for TSNE.fit_transform iteration-source validation."""
    del n_iter
    del max_iter
    return True


def witness_tsne_fit_transform_max_iter(
    n_iter: str | int,
    max_iter: int | None,
) -> int:
    """Describe the resolved TSNE._max_iter value."""
    del n_iter
    del max_iter
    return 1000


def witness_tsne_n_features_out(
    embedding: AbstractArray,
) -> int:
    """Describe the output-width property of a fitted t-SNE embedding."""
    if len(embedding.shape) != 2:
        raise ValueError("embedding must be 2D")
    if int(embedding.shape[0]) < 1 or int(embedding.shape[1]) < 1:
        raise ValueError("embedding must be nonempty")
    return int(embedding.shape[1])


def witness_tsne_pairwise_input_tag(
    metric: str,
) -> bool:
    """Describe the t-SNE pairwise-input tag."""
    del metric
    return False
