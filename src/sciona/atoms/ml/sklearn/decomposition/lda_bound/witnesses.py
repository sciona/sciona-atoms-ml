"""Ghost witnesses for LatentDirichletAllocation variational-bound helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_lda_dirichlet_loglikelihood(
    prior: float,
    distr: AbstractArray,
    dirichlet_distr: AbstractArray,
    size: int,
) -> AbstractArray:
    """Describe sklearn's scalar Dirichlet log-likelihood contribution."""
    del prior, size
    if len(distr.shape) != 2 or len(dirichlet_distr.shape) != 2:
        raise ValueError("distribution inputs must be 2D")
    if tuple(distr.shape) != tuple(dirichlet_distr.shape):
        raise ValueError("distribution inputs must share shape")
    return AbstractArray(shape=(), dtype="float64")


def witness_lda_document_log_probability_bound(
    X: AbstractArray,
    dirichlet_doc_topic: AbstractArray,
    dirichlet_component: AbstractArray,
) -> AbstractArray:
    """Describe the scalar document-word bound term from supplied expectations."""
    if len(X.shape) != 2 or len(dirichlet_doc_topic.shape) != 2 or len(dirichlet_component.shape) != 2:
        raise ValueError("all inputs must be 2D")
    if int(X.shape[0]) != int(dirichlet_doc_topic.shape[0]):
        raise ValueError("X and dirichlet_doc_topic must share the document axis")
    if int(dirichlet_doc_topic.shape[1]) != int(dirichlet_component.shape[0]):
        raise ValueError("topic axes must align")
    if int(X.shape[1]) != int(dirichlet_component.shape[1]):
        raise ValueError("word axes must align")
    return AbstractArray(shape=(), dtype="float64")


def witness_lda_apply_subsampling_ratio(
    score: float,
    *,
    total_samples: float,
    current_samples: int,
    sub_sampling: bool = False,
) -> AbstractArray:
    """Describe sklearn's optional bound rescaling under subsampling."""
    del score, total_samples, current_samples, sub_sampling
    return AbstractArray(shape=(), dtype="float64")


def witness_lda_approx_bound_from_expectations(
    X: AbstractArray,
    doc_topic_distr: AbstractArray,
    dirichlet_doc_topic: AbstractArray,
    components: AbstractArray,
    dirichlet_component: AbstractArray,
    *,
    doc_topic_prior: float,
    topic_word_prior: float,
    total_samples: float = 1.0,
    sub_sampling: bool = False,
) -> AbstractArray:
    """Describe sklearn's scalar approximate variational bound from supplied expectations."""
    del doc_topic_prior, topic_word_prior, total_samples, sub_sampling
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if len(doc_topic_distr.shape) != 2 or len(dirichlet_doc_topic.shape) != 2:
        raise ValueError("document-topic inputs must be 2D")
    if len(components.shape) != 2 or len(dirichlet_component.shape) != 2:
        raise ValueError("component inputs must be 2D")
    if tuple(doc_topic_distr.shape) != tuple(dirichlet_doc_topic.shape):
        raise ValueError("document-topic inputs must share shape")
    if tuple(components.shape) != tuple(dirichlet_component.shape):
        raise ValueError("component inputs must share shape")
    if int(X.shape[0]) != int(doc_topic_distr.shape[0]):
        raise ValueError("document counts must match")
    if int(X.shape[1]) != int(components.shape[1]):
        raise ValueError("feature counts must match")
    if int(doc_topic_distr.shape[1]) != int(components.shape[0]):
        raise ValueError("topic counts must match")
    return AbstractArray(shape=(), dtype="float64")

