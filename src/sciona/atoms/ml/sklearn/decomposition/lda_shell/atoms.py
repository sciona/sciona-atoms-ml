"""LatentDirichletAllocation helper atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_lda_doc_topic_prior,
    witness_lda_component_values,
    witness_lda_normalize_document_topics,
    witness_lda_perplexity_from_bound,
    witness_lda_perplexity_require_matching_samples,
    witness_lda_perplexity_require_matching_topics,
    witness_lda_perplexity_word_count,
    witness_lda_topic_word_prior,
)


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _optional_unit_interval(value: object) -> bool:
    return value is None or (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def _dtype_name_valid(dtype_name: object) -> bool:
    return dtype_name in {"float32", "float64"}


def _positive_finite_scalar(value: object) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(float(value))
        and float(value) > 0.0
    )


def _bound_scalar_valid(value: object) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)))


def _nonnegative_finite_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)) and np.all(array >= 0.0))


def _strictly_positive_row_sums(values: object) -> bool:
    array = np.asarray(values, dtype=np.float64)
    return bool(_nonnegative_finite_matrix(values) and np.all(np.sum(array, axis=1) > 0.0))


def _components_valid(result: object, n_components: int, n_features: int, dtype_name: str) -> bool:
    values = np.asarray(result)
    return bool(
        values.shape == (n_components, n_features)
        and values.dtype == np.dtype(dtype_name)
        and np.all(np.isfinite(values))
        and np.all(values > 0.0)
    )


def _normalized_topics_valid(result: object, doc_topic_distr: object) -> bool:
    values = np.asarray(result, dtype=np.float64)
    source = np.asarray(doc_topic_distr, dtype=np.float64)
    return bool(
        values.shape == source.shape
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and np.allclose(np.sum(values, axis=1), 1.0)
    )


@register_atom(witness_lda_doc_topic_prior)
@icontract.require(lambda n_components: _positive_int(n_components), "n_components must be a positive integer")
@icontract.require(lambda doc_topic_prior: _optional_unit_interval(doc_topic_prior), "doc_topic_prior must be None or a finite value in [0, 1]")
@icontract.ensure(lambda result: _positive_finite_scalar(result), "doc topic prior must be a finite positive scalar")
def lda_doc_topic_prior(
    n_components: int,
    *,
    doc_topic_prior: float | None = None,
) -> float:
    """Resolve sklearn's fitted doc_topic_prior_ value."""
    if doc_topic_prior is None:
        return float(1.0 / int(n_components))
    return float(doc_topic_prior)


@register_atom(witness_lda_topic_word_prior)
@icontract.require(lambda n_components: _positive_int(n_components), "n_components must be a positive integer")
@icontract.require(lambda topic_word_prior: _optional_unit_interval(topic_word_prior), "topic_word_prior must be None or a finite value in [0, 1]")
@icontract.ensure(lambda result: _positive_finite_scalar(result), "topic word prior must be a finite positive scalar")
def lda_topic_word_prior(
    n_components: int,
    *,
    topic_word_prior: float | None = None,
) -> float:
    """Resolve sklearn's fitted topic_word_prior_ value."""
    if topic_word_prior is None:
        return float(1.0 / int(n_components))
    return float(topic_word_prior)


@register_atom(witness_lda_component_values)
@icontract.require(lambda n_components: _positive_int(n_components), "n_components must be a positive integer")
@icontract.require(lambda n_features: _positive_int(n_features), "n_features must be a positive integer")
@icontract.require(lambda random_state: isinstance(random_state, int) and not isinstance(random_state, bool), "random_state must be an integer seed")
@icontract.require(lambda dtype_name: _dtype_name_valid(dtype_name), "dtype_name must be 'float32' or 'float64'")
@icontract.ensure(lambda result, n_components, n_features, dtype_name: _components_valid(result, n_components, n_features, dtype_name), "initial components must match sklearn's positive gamma sample matrix")
def lda_component_values(
    n_components: int,
    n_features: int,
    *,
    random_state: int,
    dtype_name: str = "float64",
) -> NDArray[np.float64] | NDArray[np.float32]:
    """Create the positive starting value table for a Latent Dirichlet Allocation model."""
    dtype = np.dtype(dtype_name)
    values = np.random.RandomState(random_state).gamma(
        100.0,
        0.01,
        (int(n_components), int(n_features)),
    ).astype(dtype, copy=False)
    return values


@register_atom(witness_lda_normalize_document_topics)
@icontract.require(lambda doc_topic_distr: _strictly_positive_row_sums(doc_topic_distr), "doc_topic_distr must be a finite nonnegative matrix with strictly positive row sums")
@icontract.ensure(lambda result, doc_topic_distr: _normalized_topics_valid(result, doc_topic_distr), "normalized document topics must preserve shape and sum to one per row")
def lda_normalize_document_topics(
    doc_topic_distr: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Normalize sklearn's unnormalized document-topic distribution row-wise."""
    values = np.asarray(doc_topic_distr, dtype=np.float64)
    return np.asarray(values / np.sum(values, axis=1)[:, np.newaxis], dtype=np.float64)


@register_atom(witness_lda_perplexity_require_matching_samples)
@icontract.require(lambda doc_topic_distr: _nonnegative_finite_matrix(doc_topic_distr), "doc_topic_distr must be a finite nonnegative 2D matrix")
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be a positive integer")
@icontract.ensure(lambda result, doc_topic_distr: np.array_equal(np.asarray(result, dtype=np.float64), np.asarray(doc_topic_distr, dtype=np.float64)), "validated topic matrix must preserve the supplied values")
def lda_perplexity_require_matching_samples(
    doc_topic_distr: NDArray[np.float64],
    *,
    n_samples: int,
) -> NDArray[np.float64]:
    """Require sklearn's perplexity sample-count check for a supplied topic matrix."""
    values = np.asarray(doc_topic_distr, dtype=np.float64)
    if values.shape[0] != int(n_samples):
        raise ValueError("Number of samples in X and doc_topic_distr do not match.")
    return np.asarray(values, dtype=np.float64)


@register_atom(witness_lda_perplexity_require_matching_topics)
@icontract.require(lambda doc_topic_distr: _nonnegative_finite_matrix(doc_topic_distr), "doc_topic_distr must be a finite nonnegative 2D matrix")
@icontract.require(lambda n_components: _positive_int(n_components), "n_components must be a positive integer")
@icontract.ensure(lambda result, doc_topic_distr: np.array_equal(np.asarray(result, dtype=np.float64), np.asarray(doc_topic_distr, dtype=np.float64)), "validated topic matrix must preserve the supplied values")
def lda_perplexity_require_matching_topics(
    doc_topic_distr: NDArray[np.float64],
    *,
    n_components: int,
) -> NDArray[np.float64]:
    """Require sklearn's perplexity topic-count check for a supplied topic matrix."""
    values = np.asarray(doc_topic_distr, dtype=np.float64)
    if values.shape[1] != int(n_components):
        raise ValueError("Number of topics does not match.")
    return np.asarray(values, dtype=np.float64)


@register_atom(witness_lda_perplexity_word_count)
@icontract.require(lambda total_term_count: _positive_finite_scalar(total_term_count), "total_term_count must be a finite positive scalar")
@icontract.require(lambda current_samples: _positive_int(current_samples), "current_samples must be a positive integer")
@icontract.require(lambda total_samples: _positive_finite_scalar(total_samples), "total_samples must be a finite positive scalar")
@icontract.require(lambda sub_sampling: isinstance(sub_sampling, bool), "sub_sampling must be boolean")
@icontract.ensure(lambda result: _positive_finite_scalar(result), "word count must be a finite positive scalar")
def lda_perplexity_word_count(
    total_term_count: float,
    *,
    current_samples: int,
    total_samples: float,
    sub_sampling: bool = False,
) -> float:
    """Compute sklearn's effective word count used in perplexity."""
    if sub_sampling:
        return float(total_term_count) * (float(total_samples) / float(current_samples))
    return float(total_term_count)


@register_atom(witness_lda_perplexity_from_bound)
@icontract.require(lambda bound: _bound_scalar_valid(bound), "bound must be a finite scalar")
@icontract.require(lambda word_count: _positive_finite_scalar(word_count), "word_count must be a finite positive scalar")
@icontract.ensure(lambda result: _positive_finite_scalar(result), "perplexity must be a finite positive scalar")
def lda_perplexity_from_bound(
    bound: float,
    *,
    word_count: float,
) -> float:
    """Convert sklearn's variational bound and effective word count to perplexity."""
    perword_bound = float(bound) / float(word_count)
    return float(np.exp(-1.0 * perword_bound))
