"""Ghost witnesses for sklearn LDA post-fit shell helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _matrix_shape(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    rows, cols = int(values.shape[0]), int(values.shape[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"{name} must be nonempty")
    return rows, cols


def witness_lda_unnormalized_transform_output(
    doc_topic_distr: AbstractArray,
) -> AbstractArray:
    """Describe LDA's unnormalized-transform shell output."""
    return AbstractArray(shape=_matrix_shape(doc_topic_distr, "doc_topic_distr"), dtype="float64")


def witness_lda_transform_output(
    doc_topic_distr: AbstractArray,
    *,
    normalize: bool = True,
) -> AbstractArray:
    """Describe LDA's transform shell output."""
    del normalize
    return AbstractArray(shape=_matrix_shape(doc_topic_distr, "doc_topic_distr"), dtype="float64")


def witness_lda_score_from_bound(
    bound: float,
) -> float:
    """Describe LDA's score shell output from a supplied bound."""
    del bound
    return 0.0


def witness_lda_n_features_out(
    n_components: int,
) -> int:
    """Describe LDA's transformed output width."""
    del n_components
    return 1
