"""Ghost witnesses for sklearn LDA perplexity-shell helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _matrix_shape(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    rows, cols = int(values.shape[0]), int(values.shape[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"{name} must be nonempty")
    return rows, cols


def witness_lda_perplexity_precomputed_topics(
    bound: float,
    X: AbstractArray,
    doc_topic_distr: AbstractArray,
    *,
    n_components: int,
    total_samples: float = 1.0,
    sub_sampling: bool = False,
) -> float:
    """Describe LDA perplexity from a supplied approximate bound and topic matrix."""
    del bound
    del total_samples
    del sub_sampling
    x_rows, _ = _matrix_shape(X, "X")
    topic_rows, topic_cols = _matrix_shape(doc_topic_distr, "doc_topic_distr")
    if x_rows != topic_rows:
        raise ValueError("X and doc_topic_distr must share the sample axis")
    if topic_cols != int(n_components):
        raise ValueError("doc_topic_distr topic width must match n_components")
    return 1.0


def witness_lda_fit_transform_output(
    transformed: AbstractArray,
) -> AbstractArray:
    """Describe LDA's fit_transform shell output."""
    return AbstractArray(shape=_matrix_shape(transformed, "transformed"), dtype="float64")
