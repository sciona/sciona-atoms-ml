"""LatentDirichletAllocation variational-bound helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln, logsumexp

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_lda_apply_subsampling_ratio,
    witness_lda_approx_bound_from_expectations,
    witness_lda_dirichlet_loglikelihood,
    witness_lda_document_log_probability_bound,
)


def _positive_finite_scalar(value: object) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(float(value))
        and float(value) > 0.0
    )


def _finite_scalar(value: object) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(float(value))
    )


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _nonnegative_finite_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 2
        and array.shape[0] >= 1
        and array.shape[1] >= 1
        and np.all(np.isfinite(array))
        and np.all(array >= 0.0)
    )


def _finite_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 2
        and array.shape[0] >= 1
        and array.shape[1] >= 1
        and np.all(np.isfinite(array))
    )


def _strictly_positive_matrix(values: object) -> bool:
    return bool(_nonnegative_finite_matrix(values) and np.all(np.asarray(values, dtype=np.float64) > 0.0))


def _same_shape(left: object, right: object) -> bool:
    return bool(np.asarray(left).shape == np.asarray(right).shape)


def _document_term_inputs_valid(
    X: NDArray[np.float64],
    dirichlet_doc_topic: NDArray[np.float64],
    dirichlet_component: NDArray[np.float64],
) -> bool:
    if not (
        _nonnegative_finite_matrix(X)
        and _finite_matrix(dirichlet_doc_topic)
        and _finite_matrix(dirichlet_component)
    ):
        return False
    counts = np.asarray(X, dtype=np.float64)
    doc_topic = np.asarray(dirichlet_doc_topic, dtype=np.float64)
    component = np.asarray(dirichlet_component, dtype=np.float64)
    return bool(
        counts.shape[0] == doc_topic.shape[0]
        and doc_topic.shape[1] == component.shape[0]
        and counts.shape[1] == component.shape[1]
    )


@register_atom(witness_lda_dirichlet_loglikelihood)
@icontract.require(lambda prior: _positive_finite_scalar(prior), "prior must be a finite positive scalar")
@icontract.require(lambda distr: _strictly_positive_matrix(distr), "distr must be a finite strictly positive matrix")
@icontract.require(lambda dirichlet_distr: _finite_matrix(dirichlet_distr), "dirichlet_distr must be a finite matrix")
@icontract.require(lambda distr, dirichlet_distr: _same_shape(distr, dirichlet_distr), "distr and dirichlet_distr must share shape")
@icontract.require(lambda size: _positive_int(size), "size must be a positive integer")
@icontract.ensure(lambda result: _finite_scalar(result), "Dirichlet log-likelihood contribution must be finite")
def lda_dirichlet_loglikelihood(
    prior: float,
    distr: NDArray[np.float64],
    dirichlet_distr: NDArray[np.float64],
    size: int,
) -> float:
    """Compute one scalar Dirichlet prior-minus-variational log-likelihood term."""
    values = np.asarray(distr, dtype=np.float64)
    expectations = np.asarray(dirichlet_distr, dtype=np.float64)
    score = np.sum((float(prior) - values) * expectations)
    score += np.sum(gammaln(values) - gammaln(float(prior)))
    score += np.sum(gammaln(float(prior) * int(size)) - gammaln(np.sum(values, axis=1)))
    return float(score)


@register_atom(witness_lda_document_log_probability_bound)
@icontract.require(lambda X, dirichlet_doc_topic, dirichlet_component: _document_term_inputs_valid(X, dirichlet_doc_topic, dirichlet_component), "X, dirichlet_doc_topic, and dirichlet_component must be finite compatible matrices")
@icontract.ensure(lambda result: _finite_scalar(result), "document-word bound term must be finite")
def lda_document_log_probability_bound(
    X: NDArray[np.float64],
    dirichlet_doc_topic: NDArray[np.float64],
    dirichlet_component: NDArray[np.float64],
) -> float:
    """Compute the document-word bound contribution from supplied expectations."""
    counts = np.asarray(X, dtype=np.float64)
    doc_topic = np.asarray(dirichlet_doc_topic, dtype=np.float64)
    component = np.asarray(dirichlet_component, dtype=np.float64)
    score = 0.0
    for idx_d in range(counts.shape[0]):
        ids = np.nonzero(counts[idx_d, :])[0]
        cnts = counts[idx_d, ids]
        temp = doc_topic[idx_d, :, np.newaxis] + component[:, ids]
        norm_phi = logsumexp(temp, axis=0)
        score += float(np.dot(cnts, norm_phi))
    return float(score)


@register_atom(witness_lda_apply_subsampling_ratio)
@icontract.require(lambda score: _finite_scalar(score), "score must be finite")
@icontract.require(lambda total_samples: _positive_finite_scalar(total_samples), "total_samples must be a finite positive scalar")
@icontract.require(lambda current_samples: _positive_int(current_samples), "current_samples must be a positive integer")
@icontract.require(lambda sub_sampling: isinstance(sub_sampling, bool), "sub_sampling must be boolean")
@icontract.ensure(lambda result: _finite_scalar(result), "rescaled score must be finite")
def lda_apply_subsampling_ratio(
    score: float,
    *,
    total_samples: float,
    current_samples: int,
    sub_sampling: bool = False,
) -> float:
    """Apply sklearn's optional document subsampling ratio to a bound term."""
    if sub_sampling:
        return float(score) * (float(total_samples) / float(current_samples))
    return float(score)


@register_atom(witness_lda_approx_bound_from_expectations)
@icontract.require(lambda X, doc_topic_distr, dirichlet_doc_topic, components, dirichlet_component: _document_term_inputs_valid(X, dirichlet_doc_topic, dirichlet_component) and _strictly_positive_matrix(doc_topic_distr) and _strictly_positive_matrix(components), "document counts, expectations, and positive variational parameters must be compatible")
@icontract.require(lambda doc_topic_distr, dirichlet_doc_topic: _same_shape(doc_topic_distr, dirichlet_doc_topic), "doc_topic_distr and dirichlet_doc_topic must share shape")
@icontract.require(lambda components, dirichlet_component: _same_shape(components, dirichlet_component), "components and dirichlet_component must share shape")
@icontract.require(lambda X, doc_topic_distr, components: np.asarray(X).shape[0] == np.asarray(doc_topic_distr).shape[0] and np.asarray(X).shape[1] == np.asarray(components).shape[1] and np.asarray(doc_topic_distr).shape[1] == np.asarray(components).shape[0], "document, topic, and feature axes must align")
@icontract.require(lambda doc_topic_prior: _positive_finite_scalar(doc_topic_prior), "doc_topic_prior must be finite and positive")
@icontract.require(lambda topic_word_prior: _positive_finite_scalar(topic_word_prior), "topic_word_prior must be finite and positive")
@icontract.require(lambda total_samples: _positive_finite_scalar(total_samples), "total_samples must be a finite positive scalar")
@icontract.require(lambda sub_sampling: isinstance(sub_sampling, bool), "sub_sampling must be boolean")
@icontract.ensure(lambda result: _finite_scalar(result), "approximate bound must be finite")
def lda_approx_bound_from_expectations(
    X: NDArray[np.float64],
    doc_topic_distr: NDArray[np.float64],
    dirichlet_doc_topic: NDArray[np.float64],
    components: NDArray[np.float64],
    dirichlet_component: NDArray[np.float64],
    *,
    doc_topic_prior: float,
    topic_word_prior: float,
    total_samples: float = 1.0,
    sub_sampling: bool = False,
) -> float:
    """Compute sklearn's approximate LDA variational bound from supplied expectations."""
    counts = np.asarray(X, dtype=np.float64)
    doc_topic_values = np.asarray(doc_topic_distr, dtype=np.float64)
    component_values = np.asarray(components, dtype=np.float64)
    score = lda_document_log_probability_bound(
        counts,
        np.asarray(dirichlet_doc_topic, dtype=np.float64),
        np.asarray(dirichlet_component, dtype=np.float64),
    )
    score += lda_dirichlet_loglikelihood(
        float(doc_topic_prior),
        doc_topic_values,
        np.asarray(dirichlet_doc_topic, dtype=np.float64),
        component_values.shape[0],
    )
    score = lda_apply_subsampling_ratio(
        score,
        total_samples=float(total_samples),
        current_samples=counts.shape[0],
        sub_sampling=sub_sampling,
    )
    score += lda_dirichlet_loglikelihood(
        float(topic_word_prior),
        component_values,
        np.asarray(dirichlet_component, dtype=np.float64),
        component_values.shape[1],
    )
    return float(score)

