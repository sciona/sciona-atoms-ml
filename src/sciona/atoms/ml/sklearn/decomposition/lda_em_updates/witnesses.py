"""Ghost witnesses for sklearn LDA EM-update helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _matrix_shape(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    rows, cols = int(values.shape[0]), int(values.shape[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"{name} must be nonempty")
    return rows, cols


def witness_lda_e_step_document_topic_matrix(
    doc_topic_blocks: tuple[AbstractArray, ...],
) -> AbstractArray:
    """Describe the stacked document-topic matrix returned from LDA E-step blocks."""
    if len(doc_topic_blocks) < 1:
        raise ValueError("doc_topic_blocks must be nonempty")
    total_rows = 0
    n_topics = None
    for index, block in enumerate(doc_topic_blocks):
        rows, cols = _matrix_shape(block, f"doc_topic_blocks[{index}]")
        total_rows += rows
        if n_topics is None:
            n_topics = cols
        elif cols != n_topics:
            raise ValueError("doc_topic_blocks must share the same topic width")
    assert n_topics is not None
    return AbstractArray(shape=(total_rows, n_topics), dtype="float64")


def witness_lda_e_step_sufficient_statistics(
    sstats_blocks: tuple[AbstractArray, ...],
    exp_dirichlet_component: AbstractArray,
) -> AbstractArray:
    """Describe the merged sufficient-statistics matrix returned from LDA E-step blocks."""
    target_shape = _matrix_shape(exp_dirichlet_component, "exp_dirichlet_component")
    if len(sstats_blocks) < 1:
        raise ValueError("sstats_blocks must be nonempty")
    for index, block in enumerate(sstats_blocks):
        if _matrix_shape(block, f"sstats_blocks[{index}]") != target_shape:
            raise ValueError("sstats_blocks must share exp_dirichlet_component shape")
    return AbstractArray(shape=target_shape, dtype="float64")


def witness_lda_online_update_weight(
    learning_offset: float,
    n_batch_iter: int,
    learning_decay: float,
) -> float:
    """Describe the online update weight returned from sklearn's LDA update schedule."""
    del learning_offset
    del n_batch_iter
    del learning_decay
    return 0.5


def witness_lda_online_document_ratio(
    total_samples: float,
    current_samples: int,
) -> float:
    """Describe the document ratio used by sklearn's online LDA update."""
    del total_samples
    del current_samples
    return 1.0


def witness_lda_batch_components(
    topic_word_prior: float,
    suff_stats: AbstractArray,
) -> AbstractArray:
    """Describe the batch-updated LDA components matrix."""
    del topic_word_prior
    return AbstractArray(shape=_matrix_shape(suff_stats, "suff_stats"), dtype="float64")


def witness_lda_online_components(
    previous_components: AbstractArray,
    topic_word_prior: float,
    suff_stats: AbstractArray,
    *,
    weight: float,
    doc_ratio: float,
) -> AbstractArray:
    """Describe the online-updated LDA components matrix."""
    del topic_word_prior
    del weight
    del doc_ratio
    previous_shape = _matrix_shape(previous_components, "previous_components")
    if _matrix_shape(suff_stats, "suff_stats") != previous_shape:
        raise ValueError("suff_stats must match previous_components shape")
    return AbstractArray(shape=previous_shape, dtype="float64")
