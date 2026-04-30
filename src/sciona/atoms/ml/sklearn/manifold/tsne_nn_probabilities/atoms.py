"""Nearest-neighbor t-SNE probability helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, isspmatrix_csr

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_tsne_nn_conditional_probability_matrix,
    witness_tsne_nn_distance_blocks,
    witness_tsne_nn_joint_probabilities,
)

MACHINE_EPSILON = np.finfo(np.double).eps


def _csr_nonnegative(values: object) -> bool:
    return bool(
        isspmatrix_csr(values)
        and values.shape[0] >= 2
        and values.shape[1] >= 2
        and np.all(np.isfinite(values.data))
        and np.all(values.data >= 0.0)
    )


def _distance_blocks_valid(result: object, distances: csr_matrix) -> bool:
    try:
        values = np.asarray(result, dtype=np.float32)
    except (TypeError, ValueError):
        return False
    source = distances
    return bool(
        values.ndim == 2
        and values.shape[0] == source.shape[0]
        and values.size == source.data.size
        and np.all(np.isfinite(values))
    )


def _int_vector(values: object) -> bool:
    array = np.asarray(values)
    return bool(array.ndim == 1 and np.issubdtype(array.dtype, np.integer))


def _conditional_inputs_valid(
    conditional_probabilities: object,
    indices: object,
    indptr: object,
    n_samples: int,
) -> bool:
    try:
        probs = np.asarray(conditional_probabilities, dtype=np.float64)
        idx = np.asarray(indices)
        ptr = np.asarray(indptr)
    except (TypeError, ValueError):
        return False
    return bool(
        probs.ndim == 2
        and probs.shape[0] == n_samples
        and probs.shape[1] >= 1
        and n_samples >= 2
        and np.all(np.isfinite(probs))
        and np.all(probs >= 0.0)
        and _int_vector(idx)
        and _int_vector(ptr)
        and ptr.shape == (n_samples + 1,)
        and ptr[0] == 0
        and np.all(ptr[:-1] <= ptr[1:])
        and ptr[-1] == idx.shape[0] == probs.size
        and np.all(idx >= 0)
        and np.all(idx < n_samples)
    )


def _csr_shape_valid(result: object, n_samples: int) -> bool:
    return bool(
        isspmatrix_csr(result)
        and result.shape == (n_samples, n_samples)
        and np.all(np.isfinite(result.data))
        and np.all(result.data >= 0.0)
    )


def _joint_probabilities_valid(result: object, conditional_probability_matrix: csr_matrix) -> bool:
    if not _csr_shape_valid(result, conditional_probability_matrix.shape[0]):
        return False
    values = result
    return bool(
        np.isclose(float(values.sum()), 1.0)
        and np.all(np.abs((values - values.T).data) <= 1e-12)
        and np.all(values.data <= 1.0)
    )


@register_atom(witness_tsne_nn_distance_blocks)
@icontract.require(lambda distances: _csr_nonnegative(distances), "distances must be a finite nonnegative CSR matrix with at least two samples")
@icontract.ensure(lambda result, distances: _distance_blocks_valid(result, distances), "distance blocks must reshape the CSR data into one row per sample")
def tsne_nn_distance_blocks(distances: csr_matrix) -> NDArray[np.float32]:
    """Extract the nearest-neighbor distance rows sklearn feeds into perplexity search."""
    sorted_distances = distances.copy().tocsr()
    sorted_distances.sort_indices()
    n_samples = sorted_distances.shape[0]
    blocks = sorted_distances.data.reshape(n_samples, -1)
    return blocks.astype(np.float32, copy=False)


@register_atom(witness_tsne_nn_conditional_probability_matrix)
@icontract.require(
    lambda conditional_probabilities, indices, indptr, n_samples: _conditional_inputs_valid(
        conditional_probabilities, indices, indptr, n_samples
    ),
    "conditional probabilities, indices, and indptr must define one aligned CSR row per sample",
)
@icontract.ensure(lambda result, n_samples: _csr_shape_valid(result, n_samples), "conditional-probability matrix must be a finite nonnegative CSR matrix")
def tsne_nn_conditional_probability_matrix(
    conditional_probabilities: NDArray[np.float64],
    indices: NDArray[np.int64],
    indptr: NDArray[np.int64],
    *,
    n_samples: int,
) -> csr_matrix:
    """Rebuild sklearn's sparse conditional-probability matrix from perplexity-search outputs."""
    probabilities = np.asarray(conditional_probabilities, dtype=np.float64)
    return csr_matrix(
        (probabilities.ravel(), np.asarray(indices, dtype=np.int64), np.asarray(indptr, dtype=np.int64)),
        shape=(n_samples, n_samples),
    )


@register_atom(witness_tsne_nn_joint_probabilities)
@icontract.require(lambda conditional_probability_matrix: _csr_nonnegative(conditional_probability_matrix), "conditional_probability_matrix must be a finite nonnegative CSR square matrix")
@icontract.ensure(
    lambda result, conditional_probability_matrix: _joint_probabilities_valid(result, conditional_probability_matrix),
    "joint probabilities must be symmetric, finite, nonnegative, and sum to one",
)
def tsne_nn_joint_probabilities(
    conditional_probability_matrix: csr_matrix,
) -> csr_matrix:
    """Symmetrize and normalize nearest-neighbor t-SNE conditional probabilities."""
    joint = conditional_probability_matrix + conditional_probability_matrix.T
    sum_joint = np.maximum(joint.sum(), MACHINE_EPSILON)
    joint = joint / sum_joint
    return joint.tocsr()
