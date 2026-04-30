"""LatentDirichletAllocation perplexity-shell helper atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_lda_fit_transform_output,
    witness_lda_perplexity_precomputed_topics,
)


def _positive_finite_scalar(value: object) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(float(value))
        and float(value) > 0.0
    )


def _positive_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 2
        and array.shape[0] >= 1
        and array.shape[1] >= 1
        and np.all(np.isfinite(array))
        and np.all(array > 0.0)
    )


def _count_matrix(values: object) -> bool:
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


def _normalized_or_positive_matrix_valid(result: object, source: object) -> bool:
    values = np.asarray(result, dtype=np.float64)
    source_values = np.asarray(source, dtype=np.float64)
    return bool(values.shape == source_values.shape and np.all(np.isfinite(values)) and np.all(values > 0.0))


@register_atom(witness_lda_perplexity_precomputed_topics)
@icontract.require(lambda bound: isinstance(bound, (int, float)) and not isinstance(bound, bool) and np.isfinite(float(bound)), "bound must be a finite scalar")
@icontract.require(lambda X: _count_matrix(X), "X must be a finite nonnegative 2D matrix")
@icontract.require(lambda doc_topic_distr: _positive_matrix(doc_topic_distr), "doc_topic_distr must be a finite strictly positive 2D matrix")
@icontract.require(lambda n_components: isinstance(n_components, int) and not isinstance(n_components, bool) and n_components >= 1, "n_components must be a positive integer")
@icontract.require(lambda total_samples: _positive_finite_scalar(total_samples), "total_samples must be a finite positive scalar")
@icontract.require(lambda sub_sampling: isinstance(sub_sampling, bool), "sub_sampling must be boolean")
@icontract.ensure(lambda result: _positive_finite_scalar(result), "perplexity must be a finite positive scalar")
def lda_perplexity_precomputed_topics(
    bound: float,
    X: NDArray[np.float64],
    doc_topic_distr: NDArray[np.float64],
    *,
    n_components: int,
    total_samples: float = 1.0,
    sub_sampling: bool = False,
) -> float:
    """Compute LDA perplexity from a supplied approximate bound and prevalidated topic matrix."""
    counts = np.asarray(X, dtype=np.float64)
    topics = np.asarray(doc_topic_distr, dtype=np.float64)
    if topics.shape[0] != counts.shape[0]:
        raise ValueError("Number of samples in X and doc_topic_distr do not match.")
    if topics.shape[1] != int(n_components):
        raise ValueError("Number of topics does not match.")

    current_samples = counts.shape[0]
    if sub_sampling:
        word_cnt = float(np.sum(counts)) * (float(total_samples) / float(current_samples))
    else:
        word_cnt = float(np.sum(counts))
    perword_bound = float(bound) / float(word_cnt)
    return float(np.exp(-1.0 * perword_bound))


@register_atom(witness_lda_fit_transform_output)
@icontract.require(lambda transformed: _positive_matrix(transformed), "transformed must be a finite strictly positive 2D matrix")
@icontract.ensure(lambda result, transformed: _normalized_or_positive_matrix_valid(result, transformed), "fit_transform output must preserve the supplied transformed matrix shape and finite values")
def lda_fit_transform_output(
    transformed: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return LDA's fit_transform shell result from a supplied transform output."""
    return np.asarray(transformed, dtype=np.float64)
