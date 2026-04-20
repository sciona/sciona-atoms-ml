"""Manifold-learning atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_symmetric
from sklearn.utils.extmath import svd_flip
from sklearn.utils.validation import check_array

from sciona.ghost.registry import register_atom

from .state_models import ClassicalMDSState
from .witnesses import (
    witness_classical_mds_dissimilarity_matrix,
    witness_classical_mds_double_center,
    witness_classical_mds_fit,
)


def _matrix_2d(X: NDArray[np.float64]) -> bool:
    values = np.asarray(X)
    return bool(values.ndim == 2)


def _square_matrix(matrix: NDArray[np.float64]) -> bool:
    values = np.asarray(matrix)
    return bool(values.ndim == 2 and values.shape[0] == values.shape[1])


def _finite_matrix(matrix: NDArray[np.float64]) -> bool:
    try:
        values = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(np.all(np.isfinite(values)))


def _metric_valid(metric: str) -> bool:
    return metric in {"euclidean", "precomputed"}


def _metric_params_valid(metric_params: None) -> bool:
    return metric_params is None


def _precomputed_shape_valid(X: NDArray[np.float64], metric: str) -> bool:
    return bool(metric != "precomputed" or _square_matrix(X))


def _n_components_valid(n_components: int, X: NDArray[np.float64]) -> bool:
    values = np.asarray(X)
    return bool(isinstance(n_components, int) and not isinstance(n_components, bool) and values.ndim == 2 and 1 <= n_components <= values.shape[0])


def _dissimilarity_valid(result: NDArray[np.float64], X: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    n_samples = np.asarray(X).shape[0]
    return bool(values.shape == (n_samples, n_samples) and np.all(np.isfinite(values)) and np.allclose(values, values.T))


def _centered_matrix_valid(result: NDArray[np.float64], dissimilarity_matrix: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    source = np.asarray(dissimilarity_matrix)
    return bool(
        values.shape == source.shape
        and np.all(np.isfinite(values))
        and np.allclose(values, values.T)
        and np.allclose(np.mean(values, axis=0), 0.0)
        and np.allclose(np.mean(values, axis=1), 0.0)
    )


def _classical_mds_state_valid(state: ClassicalMDSState) -> bool:
    n_samples = state.dissimilarity_matrix.shape[0]
    return bool(
        state.embedding.shape == (n_samples, state.n_components)
        and state.dissimilarity_matrix.shape == (n_samples, n_samples)
        and state.eigenvalues.shape == (state.n_components,)
        and state.n_components >= 1
        and state.n_features_in >= 1
        and _metric_valid(state.metric)
        and np.all(np.isfinite(state.embedding))
        and np.all(np.isfinite(state.dissimilarity_matrix))
        and np.all(np.isfinite(state.eigenvalues))
        and np.allclose(state.dissimilarity_matrix, state.dissimilarity_matrix.T)
        and np.all(state.eigenvalues >= -1e-10)
    )


@register_atom(witness_classical_mds_dissimilarity_matrix)
@icontract.require(lambda X: _matrix_2d(X), "X must be a two-dimensional matrix")
@icontract.require(lambda X: _finite_matrix(X), "X must contain only finite values")
@icontract.require(lambda metric: _metric_valid(metric), "metric must be euclidean or precomputed")
@icontract.require(lambda metric_params: _metric_params_valid(metric_params), "metric_params are outside this atom scope")
@icontract.require(lambda X, metric: _precomputed_shape_valid(X, metric), "precomputed dissimilarities must be square")
@icontract.ensure(lambda result, X: _dissimilarity_valid(result, X), "dissimilarities must be finite symmetric sample-by-sample distances")
def classical_mds_dissimilarity_matrix(
    X: NDArray[np.float64],
    *,
    metric: str = "euclidean",
    metric_params: None = None,
) -> NDArray[np.float64]:
    """Compute the dense dissimilarity matrix used by classical MDS."""
    del metric_params
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    if metric == "precomputed":
        return np.asarray(check_symmetric(checked_x, raise_exception=True), dtype=np.float64)
    return np.asarray(pairwise_distances(checked_x, metric="euclidean"), dtype=np.float64)


@register_atom(witness_classical_mds_double_center)
@icontract.require(lambda dissimilarity_matrix: _square_matrix(dissimilarity_matrix), "dissimilarity_matrix must be square")
@icontract.require(lambda dissimilarity_matrix: _finite_matrix(dissimilarity_matrix), "dissimilarity_matrix must contain only finite values")
@icontract.ensure(lambda result, dissimilarity_matrix: _centered_matrix_valid(result, dissimilarity_matrix), "double-centered matrix must be finite and mean centered")
def classical_mds_double_center(dissimilarity_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Center squared distances so each row and column has mean zero."""
    checked = check_array(dissimilarity_matrix, dtype=np.float64, ensure_2d=True)
    B = checked**2
    B = B.astype(np.float64)
    B -= np.mean(B, axis=0)
    B -= np.mean(B, axis=1, keepdims=True)
    B *= -0.5
    return B


@register_atom(witness_classical_mds_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be a two-dimensional matrix")
@icontract.require(lambda X: _finite_matrix(X), "X must contain only finite values")
@icontract.require(lambda n_components, X: _n_components_valid(n_components, X), "n_components must be between one and sample count")
@icontract.require(lambda metric: _metric_valid(metric), "metric must be euclidean or precomputed")
@icontract.require(lambda metric_params: _metric_params_valid(metric_params), "metric_params are outside this atom scope")
@icontract.require(lambda X, metric: _precomputed_shape_valid(X, metric), "precomputed dissimilarities must be square")
@icontract.ensure(lambda result: _classical_mds_state_valid(result), "Classical MDS state must contain finite embedding coordinates")
def classical_mds_fit(
    X: NDArray[np.float64],
    *,
    n_components: int = 2,
    metric: str = "euclidean",
    metric_params: None = None,
) -> ClassicalMDSState:
    """Fit low-dimensional coordinates from centered pairwise distances."""
    dissimilarity_matrix = classical_mds_dissimilarity_matrix(X, metric=metric, metric_params=metric_params)
    centered = classical_mds_double_center(dissimilarity_matrix)

    eigenvalues, eigenvectors = linalg.eigh(centered)
    eigenvalues = eigenvalues[::-1][:n_components]
    eigenvectors = eigenvectors[:, ::-1][:, :n_components]
    eigenvectors, _ = svd_flip(eigenvectors, None)
    embedding = np.sqrt(eigenvalues) * eigenvectors

    return ClassicalMDSState(
        embedding=np.asarray(embedding, dtype=np.float64),
        dissimilarity_matrix=dissimilarity_matrix,
        eigenvalues=np.asarray(eigenvalues, dtype=np.float64),
        n_components=n_components,
        metric=metric,
        n_features_in=int(np.asarray(X).shape[1]),
    )
