"""LatentDirichletAllocation post-fit shell helper atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_lda_n_features_out,
    witness_lda_score_from_bound,
    witness_lda_transform_output,
    witness_lda_unnormalized_transform_output,
)


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _finite_scalar(value: object) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(float(value))
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


def _normalized_matrix(values: object) -> bool:
    if not _positive_matrix(values):
        return False
    array = np.asarray(values, dtype=np.float64)
    return bool(np.allclose(np.sum(array, axis=1), 1.0))


@register_atom(witness_lda_unnormalized_transform_output)
@icontract.require(lambda doc_topic_distr: _positive_matrix(doc_topic_distr), "doc_topic_distr must be a finite strictly positive 2D matrix")
@icontract.ensure(
    lambda result, doc_topic_distr: np.array_equal(np.asarray(result, dtype=np.float64), np.asarray(doc_topic_distr, dtype=np.float64)),
    "unnormalized transform output must preserve the supplied document-topic distribution",
)
def lda_unnormalized_transform_output(
    doc_topic_distr: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return sklearn's unnormalized-transform shell result from a supplied E-step document-topic matrix."""
    return np.asarray(doc_topic_distr, dtype=np.float64)


@register_atom(witness_lda_transform_output)
@icontract.require(lambda doc_topic_distr: _positive_matrix(doc_topic_distr), "doc_topic_distr must be a finite strictly positive 2D matrix")
@icontract.require(lambda normalize: isinstance(normalize, bool), "normalize must be boolean")
@icontract.ensure(
    lambda result, doc_topic_distr, normalize: (
        _normalized_matrix(result) if normalize else np.array_equal(np.asarray(result, dtype=np.float64), np.asarray(doc_topic_distr, dtype=np.float64))
    ),
    "transform output must either normalize rows to one or preserve the supplied matrix when normalize=False",
)
def lda_transform_output(
    doc_topic_distr: NDArray[np.float64],
    *,
    normalize: bool = True,
) -> NDArray[np.float64]:
    """Apply sklearn's transform-time optional normalization to a supplied unnormalized document-topic matrix."""
    values = np.asarray(doc_topic_distr, dtype=np.float64)
    if not normalize:
        return np.asarray(values, dtype=np.float64)
    return np.asarray(values / np.sum(values, axis=1)[:, np.newaxis], dtype=np.float64)


@register_atom(witness_lda_score_from_bound)
@icontract.require(lambda bound: _finite_scalar(bound), "bound must be a finite scalar")
@icontract.ensure(lambda result: _finite_scalar(result), "score must be a finite scalar")
def lda_score_from_bound(
    bound: float,
) -> float:
    """Return sklearn's score shell result from a supplied approximate bound."""
    return float(bound)


@register_atom(witness_lda_n_features_out)
@icontract.require(lambda n_components: _positive_int(n_components), "n_components must be a positive integer")
@icontract.ensure(lambda result: _positive_int(result), "_n_features_out must be a positive integer")
def lda_n_features_out(
    n_components: int,
) -> int:
    """Return sklearn's transformed output width for a fitted LatentDirichletAllocation model."""
    return int(n_components)
