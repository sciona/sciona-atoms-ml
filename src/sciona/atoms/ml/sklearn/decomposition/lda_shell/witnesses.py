"""Ghost witnesses for LatentDirichletAllocation shell helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_lda_doc_topic_prior(
    n_components: int,
    *,
    doc_topic_prior: float | None = None,
) -> AbstractArray:
    """Describe sklearn's resolved doc_topic_prior_ scalar."""
    del doc_topic_prior
    if n_components < 1:
        raise ValueError("n_components must be positive")
    return AbstractArray(shape=(), dtype="float64", min_val=0.0)


def witness_lda_topic_word_prior(
    n_components: int,
    *,
    topic_word_prior: float | None = None,
) -> AbstractArray:
    """Describe sklearn's resolved topic_word_prior_ scalar."""
    del topic_word_prior
    if n_components < 1:
        raise ValueError("n_components must be positive")
    return AbstractArray(shape=(), dtype="float64", min_val=0.0)


def witness_lda_component_values(
    n_components: int,
    n_features: int,
    *,
    random_state: int,
    dtype_name: str = "float64",
) -> AbstractArray:
    """Describe the positive starting value table for a Latent Dirichlet Allocation model."""
    del random_state
    if n_components < 1 or n_features < 1:
        raise ValueError("n_components and n_features must be positive")
    if dtype_name not in {"float32", "float64"}:
        raise ValueError("dtype_name must be float32 or float64")
    return AbstractArray(shape=(n_components, n_features), dtype=dtype_name)


def witness_lda_normalize_document_topics(
    doc_topic_distr: AbstractArray,
) -> AbstractArray:
    """Describe sklearn's normalized document-topic matrix."""
    if len(doc_topic_distr.shape) != 2:
        raise ValueError("doc_topic_distr must be 2D")
    rows = int(doc_topic_distr.shape[0])
    cols = int(doc_topic_distr.shape[1])
    if rows < 1 or cols < 1:
        raise ValueError("doc_topic_distr must be nonempty")
    return AbstractArray(shape=(rows, cols), dtype="float64")


def witness_lda_perplexity_require_matching_samples(
    doc_topic_distr: AbstractArray,
    *,
    n_samples: int,
) -> AbstractArray:
    """Describe a perplexity doc-topic matrix after sample-count validation."""
    if len(doc_topic_distr.shape) != 2:
        raise ValueError("doc_topic_distr must be 2D")
    if int(doc_topic_distr.shape[0]) < 1 or int(doc_topic_distr.shape[1]) < 1:
        raise ValueError("doc_topic_distr must be nonempty")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    return AbstractArray(shape=(int(doc_topic_distr.shape[0]), int(doc_topic_distr.shape[1])), dtype="float64")


def witness_lda_perplexity_require_matching_topics(
    doc_topic_distr: AbstractArray,
    *,
    n_components: int,
) -> AbstractArray:
    """Describe a perplexity doc-topic matrix after topic-count validation."""
    if len(doc_topic_distr.shape) != 2:
        raise ValueError("doc_topic_distr must be 2D")
    if int(doc_topic_distr.shape[0]) < 1 or int(doc_topic_distr.shape[1]) < 1:
        raise ValueError("doc_topic_distr must be nonempty")
    if n_components < 1:
        raise ValueError("n_components must be positive")
    return AbstractArray(shape=(int(doc_topic_distr.shape[0]), int(doc_topic_distr.shape[1])), dtype="float64")


def witness_lda_perplexity_word_count(
    total_term_count: float,
    *,
    current_samples: int,
    total_samples: float,
    sub_sampling: bool = False,
) -> AbstractArray:
    """Describe sklearn's effective perplexity word-count scalar."""
    del total_term_count, current_samples, total_samples, sub_sampling
    return AbstractArray(shape=(), dtype="float64", min_val=0.0)


def witness_lda_perplexity_from_bound(
    bound: float,
    *,
    word_count: float,
) -> AbstractArray:
    """Describe sklearn's perplexity scalar from a bound and word count."""
    del bound, word_count
    return AbstractArray(shape=(), dtype="float64", min_val=0.0)
