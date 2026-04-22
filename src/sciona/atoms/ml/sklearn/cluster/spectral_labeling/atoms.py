"""Deterministic spectral-clustering label assignment atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import LinAlgError, qr, svd
from scipy.sparse import csc_matrix
from sklearn.utils import check_random_state

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_spectral_cluster_qr_labels,
    witness_spectral_discretize_labels,
)


def _embedding_valid(vectors: NDArray[np.float64]) -> bool:
    try:
        values = np.asarray(vectors, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1 or values.shape[0] < values.shape[1]:
        return False
    return bool(np.all(np.isfinite(values)))


def _discretize_embedding_valid(vectors: NDArray[np.float64]) -> bool:
    if not _embedding_valid(vectors):
        return False
    values = np.asarray(vectors, dtype=np.float64)
    if np.any(np.linalg.norm(values, axis=0) == 0.0):
        return False
    normalized = values.copy()
    norm_ones = np.sqrt(values.shape[0])
    for i in range(normalized.shape[1]):
        normalized[:, i] = (normalized[:, i] / np.linalg.norm(normalized[:, i])) * norm_ones
        if normalized[0, i] != 0:
            normalized[:, i] = -1.0 * normalized[:, i] * np.sign(normalized[0, i])
    return bool(np.all(np.linalg.norm(normalized, axis=1) > 0.0))


def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _random_state_valid(value: int | None) -> bool:
    return value is None or (isinstance(value, int) and not isinstance(value, bool) and value >= 0)


def _labels_valid(result: NDArray[np.int64], vectors: NDArray[np.float64]) -> bool:
    labels = np.asarray(result)
    values = np.asarray(vectors)
    return bool(
        labels.shape == (values.shape[0],)
        and np.issubdtype(labels.dtype, np.integer)
        and np.all(labels >= 0)
        and np.all(labels < values.shape[1])
    )


@register_atom(witness_spectral_cluster_qr_labels)
@icontract.require(lambda vectors: _embedding_valid(vectors), "vectors must be a finite nonempty embedding with samples >= components")
@icontract.ensure(lambda result, vectors: _labels_valid(result, vectors), "labels must assign one component index per sample")
def spectral_cluster_qr_labels(vectors: NDArray[np.float64]) -> NDArray[np.int64]:
    """Assign spectral-clustering labels with sklearn's direct matrix method."""
    values = np.asarray(vectors, dtype=np.float64)
    n_components = values.shape[1]
    _, _, pivots = qr(values.T, pivoting=True)
    u_t, _, v_h = svd(values[pivots[:n_components], :].T)
    aligned = np.abs(np.dot(values, np.dot(u_t, v_h.conj())))
    return np.asarray(aligned.argmax(axis=1), dtype=np.int64)


@register_atom(witness_spectral_discretize_labels)
@icontract.require(lambda vectors: _discretize_embedding_valid(vectors), "vectors must be finite with nonzero columns and rows")
@icontract.require(lambda max_svd_restarts: _positive_int(max_svd_restarts), "max_svd_restarts must be a positive integer")
@icontract.require(lambda n_iter_max: _positive_int(n_iter_max), "n_iter_max must be a positive integer")
@icontract.require(lambda random_state: _random_state_valid(random_state), "random_state must be None or a nonnegative integer")
@icontract.ensure(lambda result, vectors: _labels_valid(result, vectors), "labels must assign one component index per sample")
def spectral_discretize_labels(
    vectors: NDArray[np.float64],
    *,
    max_svd_restarts: int = 30,
    n_iter_max: int = 20,
    random_state: int | None = None,
) -> NDArray[np.int64]:
    """Assign spectral-clustering labels with sklearn's Yu-Shi discretization."""
    rng = check_random_state(random_state)
    values = np.asarray(vectors, dtype=np.float64).copy()
    eps = np.finfo(float).eps
    n_samples, n_components = values.shape

    norm_ones = np.sqrt(n_samples)
    for i in range(values.shape[1]):
        values[:, i] = (values[:, i] / np.linalg.norm(values[:, i])) * norm_ones
        if values[0, i] != 0:
            values[:, i] = -1.0 * values[:, i] * np.sign(values[0, i])

    values = values / np.sqrt((values**2).sum(axis=1))[:, np.newaxis]

    svd_restarts = 0
    has_converged = False
    labels = np.zeros(n_samples, dtype=np.int64)

    while (svd_restarts < int(max_svd_restarts)) and not has_converged:
        rotation = np.zeros((n_components, n_components), dtype=np.float64)
        rotation[:, 0] = values[rng.randint(n_samples), :].T

        c = np.zeros(n_samples, dtype=np.float64)
        for j in range(1, n_components):
            c += np.abs(np.dot(values, rotation[:, j - 1]))
            rotation[:, j] = values[c.argmin(), :].T

        last_objective_value = 0.0
        n_iter = 0

        while not has_converged:
            n_iter += 1
            t_discrete = np.dot(values, rotation)
            labels = np.asarray(t_discrete.argmax(axis=1), dtype=np.int64)
            vectors_discrete = csc_matrix(
                (np.ones(len(labels)), (np.arange(0, n_samples), labels)),
                shape=(n_samples, n_components),
            )
            t_svd = vectors_discrete.T @ values

            try:
                u, singular_values, v_h = np.linalg.svd(t_svd)
            except LinAlgError:
                svd_restarts += 1
                break

            ncut_value = 2.0 * (n_samples - singular_values.sum())
            if (abs(ncut_value - last_objective_value) < eps) or (n_iter > int(n_iter_max)):
                has_converged = True
            else:
                last_objective_value = ncut_value
                rotation = np.dot(v_h.T, u.T)

    if not has_converged:
        raise LinAlgError("SVD did not converge")
    return np.asarray(labels, dtype=np.int64)
