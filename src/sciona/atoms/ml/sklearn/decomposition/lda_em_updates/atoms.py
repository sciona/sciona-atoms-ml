"""LatentDirichletAllocation EM-update helper atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_lda_batch_components,
    witness_lda_e_step_document_topic_matrix,
    witness_lda_e_step_sufficient_statistics,
    witness_lda_online_components,
    witness_lda_online_document_ratio,
    witness_lda_online_update_weight,
)


Matrix = NDArray[np.float64]
MatrixTuple = tuple[Matrix, ...]


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _positive_finite_scalar(value: object) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(float(value))
        and float(value) > 0.0
    )


def _weight_valid(value: object) -> bool:
    return bool(_positive_finite_scalar(value) and float(value) <= 1.0)


def _finite_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _nonnegative_matrix(values: object) -> bool:
    if not _finite_matrix(values):
        return False
    array = np.asarray(values, dtype=np.float64)
    return bool(np.all(array >= 0.0))


def _positive_matrix(values: object) -> bool:
    if not _finite_matrix(values):
        return False
    array = np.asarray(values, dtype=np.float64)
    return bool(np.all(array > 0.0))


def _doc_topic_blocks_valid(doc_topic_blocks: object) -> bool:
    if not isinstance(doc_topic_blocks, tuple) or len(doc_topic_blocks) < 1:
        return False
    blocks = tuple(np.asarray(block, dtype=np.float64) for block in doc_topic_blocks)
    if not all(_positive_matrix(block) for block in blocks):
        return False
    n_topics = blocks[0].shape[1]
    return all(block.shape[1] == n_topics for block in blocks)


def _sstats_blocks_valid(sstats_blocks: object) -> bool:
    if not isinstance(sstats_blocks, tuple) or len(sstats_blocks) < 1:
        return False
    blocks = tuple(np.asarray(block, dtype=np.float64) for block in sstats_blocks)
    if not all(_nonnegative_matrix(block) for block in blocks):
        return False
    shape = blocks[0].shape
    return all(block.shape == shape for block in blocks)


def _document_topic_matrix_valid(result: object, doc_topic_blocks: object) -> bool:
    if not isinstance(doc_topic_blocks, tuple):
        return False
    stacked = np.asarray(result, dtype=np.float64)
    blocks = tuple(np.asarray(block, dtype=np.float64) for block in doc_topic_blocks)
    return bool(
        stacked.shape == (sum(block.shape[0] for block in blocks), blocks[0].shape[1])
        and np.all(np.isfinite(stacked))
        and np.all(stacked > 0.0)
    )


def _same_shape_matrix_valid(result: object, reference: object) -> bool:
    values = np.asarray(result, dtype=np.float64)
    source = np.asarray(reference, dtype=np.float64)
    return bool(values.shape == source.shape and np.all(np.isfinite(values)))


@register_atom(witness_lda_e_step_document_topic_matrix)
@icontract.require(
    lambda doc_topic_blocks: _doc_topic_blocks_valid(doc_topic_blocks),
    "doc_topic_blocks must be a nonempty tuple of finite positive matrices with a shared topic dimension",
)
@icontract.ensure(
    lambda result, doc_topic_blocks: _document_topic_matrix_valid(result, doc_topic_blocks),
    "merged document-topic matrix must stack all blocks row-wise with the shared topic width",
)
def lda_e_step_document_topic_matrix(
    doc_topic_blocks: MatrixTuple,
) -> Matrix:
    """Stack sklearn's per-job E-step document-topic blocks."""
    return np.asarray(np.vstack(doc_topic_blocks), dtype=np.float64)


@register_atom(witness_lda_e_step_sufficient_statistics)
@icontract.require(
    lambda sstats_blocks: _sstats_blocks_valid(sstats_blocks),
    "sstats_blocks must be a nonempty tuple of finite nonnegative matrices with a shared shape",
)
@icontract.require(
    lambda exp_dirichlet_component: _positive_matrix(exp_dirichlet_component),
    "exp_dirichlet_component must be a finite positive matrix",
)
@icontract.require(
    lambda sstats_blocks, exp_dirichlet_component: all(
        np.asarray(block, dtype=np.float64).shape == np.asarray(exp_dirichlet_component, dtype=np.float64).shape
        for block in sstats_blocks
    ),
    "sstats_blocks and exp_dirichlet_component must share the same shape",
)
@icontract.ensure(
    lambda result, exp_dirichlet_component: _same_shape_matrix_valid(result, exp_dirichlet_component),
    "merged sufficient statistics must match exp_dirichlet_component shape",
)
def lda_e_step_sufficient_statistics(
    sstats_blocks: MatrixTuple,
    exp_dirichlet_component: Matrix,
) -> Matrix:
    """Finish sklearn's E-step sufficient-statistics merge from per-job blocks."""
    merged = np.sum(np.asarray(sstats_blocks, dtype=np.float64), axis=0)
    return np.asarray(merged * np.asarray(exp_dirichlet_component, dtype=np.float64), dtype=np.float64)


@register_atom(witness_lda_online_update_weight)
@icontract.require(lambda learning_offset: _positive_finite_scalar(learning_offset), "learning_offset must be a finite positive scalar")
@icontract.require(lambda n_batch_iter: _positive_int(n_batch_iter), "n_batch_iter must be a positive integer")
@icontract.require(
    lambda learning_decay: isinstance(learning_decay, (int, float))
    and not isinstance(learning_decay, bool)
    and np.isfinite(float(learning_decay))
    and 0.0 <= float(learning_decay) <= 1.0,
    "learning_decay must be a finite scalar in [0, 1]",
)
@icontract.ensure(lambda result: _weight_valid(result), "online update weight must be a finite scalar in (0, 1]")
def lda_online_update_weight(
    learning_offset: float,
    n_batch_iter: int,
    learning_decay: float,
) -> float:
    """Compute sklearn's online-EM update weight."""
    return float(np.power(float(learning_offset) + int(n_batch_iter), -float(learning_decay)))


@register_atom(witness_lda_online_document_ratio)
@icontract.require(lambda total_samples: _positive_finite_scalar(total_samples), "total_samples must be a finite positive scalar")
@icontract.require(lambda current_samples: _positive_int(current_samples), "current_samples must be a positive integer")
@icontract.ensure(lambda result: _positive_finite_scalar(result), "document ratio must be a finite positive scalar")
def lda_online_document_ratio(
    total_samples: float,
    current_samples: int,
) -> float:
    """Compute sklearn's online-EM document subsampling ratio."""
    return float(total_samples) / float(current_samples)


@register_atom(witness_lda_batch_components)
@icontract.require(lambda topic_word_prior: _positive_finite_scalar(topic_word_prior), "topic_word_prior must be a finite positive scalar")
@icontract.require(lambda suff_stats: _nonnegative_matrix(suff_stats), "suff_stats must be a finite nonnegative matrix")
@icontract.ensure(lambda result, suff_stats: _same_shape_matrix_valid(result, suff_stats), "batch components must preserve the sufficient-statistics shape")
def lda_batch_components(
    topic_word_prior: float,
    suff_stats: Matrix,
) -> Matrix:
    """Apply sklearn's batch M-step component update from sufficient statistics."""
    return np.asarray(float(topic_word_prior) + np.asarray(suff_stats, dtype=np.float64), dtype=np.float64)


@register_atom(witness_lda_online_components)
@icontract.require(lambda previous_components: _positive_matrix(previous_components), "previous_components must be a finite positive matrix")
@icontract.require(lambda topic_word_prior: _positive_finite_scalar(topic_word_prior), "topic_word_prior must be a finite positive scalar")
@icontract.require(lambda suff_stats: _nonnegative_matrix(suff_stats), "suff_stats must be a finite nonnegative matrix")
@icontract.require(lambda previous_components, suff_stats: np.asarray(previous_components).shape == np.asarray(suff_stats).shape, "previous_components and suff_stats must share the same shape")
@icontract.require(lambda weight: _weight_valid(weight), "weight must be a finite scalar in (0, 1]")
@icontract.require(lambda doc_ratio: _positive_finite_scalar(doc_ratio), "doc_ratio must be a finite positive scalar")
@icontract.ensure(lambda result, previous_components: _same_shape_matrix_valid(result, previous_components), "online components must preserve the previous component shape")
def lda_online_components(
    previous_components: Matrix,
    topic_word_prior: float,
    suff_stats: Matrix,
    *,
    weight: float,
    doc_ratio: float,
) -> Matrix:
    """Apply sklearn's online M-step component update from prior sufficient statistics and weight."""
    components = np.asarray(previous_components, dtype=np.float64)
    scaled_stats = float(topic_word_prior) + float(doc_ratio) * np.asarray(suff_stats, dtype=np.float64)
    return np.asarray(components * (1.0 - float(weight)) + float(weight) * scaled_stats, dtype=np.float64)
