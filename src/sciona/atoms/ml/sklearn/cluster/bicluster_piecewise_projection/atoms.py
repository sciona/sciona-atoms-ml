"""Spectral biclustering piecewise-selection and projection atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_bicluster_piecewise_residual_norms,
    witness_bicluster_piecewise_vector,
    witness_bicluster_project_dense,
    witness_bicluster_select_best_piecewise_vectors,
)


def _finite_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _nonnegative_int_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.int64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(array >= 0))


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


def _piecewise_inputs_valid(centroids: NDArray[np.float64], labels: NDArray[np.int64]) -> bool:
    if not (_finite_matrix(centroids) and _nonnegative_int_vector(labels)):
        return False
    centroid_values = np.asarray(centroids, dtype=np.float64)
    label_values = np.asarray(labels, dtype=np.int64)
    return bool(label_values.max(initial=0) < centroid_values.shape[0])


def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _piecewise_vector_valid(result: NDArray[np.float64], centroids: NDArray[np.float64], labels: NDArray[np.int64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    centroid_values = np.asarray(centroids, dtype=np.float64)
    label_values = np.asarray(labels, dtype=np.int64)
    expected = centroid_values[label_values].ravel()
    return bool(values.shape == expected.shape and np.array_equal(values, expected))


def _residual_inputs_valid(vectors: NDArray[np.float64], piecewise_vectors: NDArray[np.float64]) -> bool:
    if not (_finite_matrix(vectors) and _finite_matrix(piecewise_vectors)):
        return False
    return bool(np.asarray(vectors, dtype=np.float64).shape == np.asarray(piecewise_vectors, dtype=np.float64).shape)


def _residual_norms_valid(result: NDArray[np.float64], vectors: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    source = np.asarray(vectors, dtype=np.float64)
    return bool(values.shape == (source.shape[0],) and np.all(np.isfinite(values)) and np.all(values >= 0.0))


def _selection_inputs_valid(vectors: NDArray[np.float64], residual_norms: NDArray[np.float64], n_best: int) -> bool:
    if not (_finite_matrix(vectors) and _finite_vector(residual_norms) and _positive_int(n_best)):
        return False
    vector_values = np.asarray(vectors, dtype=np.float64)
    residual_values = np.asarray(residual_norms, dtype=np.float64)
    return bool(residual_values.shape[0] == vector_values.shape[0] and int(n_best) <= vector_values.shape[0])


def _selected_vectors_valid(result: NDArray[np.float64], vectors: NDArray[np.float64], residual_norms: NDArray[np.float64], n_best: int) -> bool:
    values = np.asarray(result, dtype=np.float64)
    vector_values = np.asarray(vectors, dtype=np.float64)
    order = np.argsort(np.asarray(residual_norms, dtype=np.float64))[: int(n_best)]
    expected = vector_values[order]
    return bool(values.shape == expected.shape and np.array_equal(values, expected))


def _project_inputs_valid(data: NDArray[np.float64], vectors: NDArray[np.float64]) -> bool:
    if not (_finite_matrix(data) and _finite_matrix(vectors)):
        return False
    data_values = np.asarray(data, dtype=np.float64)
    vector_values = np.asarray(vectors, dtype=np.float64)
    return bool(data_values.shape[1] == vector_values.shape[0])


def _projected_result_valid(result: NDArray[np.float64], data: NDArray[np.float64], vectors: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    data_values = np.asarray(data, dtype=np.float64)
    vector_values = np.asarray(vectors, dtype=np.float64)
    return bool(values.shape == (data_values.shape[0], vector_values.shape[1]) and np.all(np.isfinite(values)))


@register_atom(witness_bicluster_piecewise_vector)
@icontract.require(lambda centroids, labels: _piecewise_inputs_valid(centroids, labels), "centroids must be finite, labels must be nonnegative integers, and labels must index existing centroid rows")
@icontract.ensure(lambda result, centroids, labels: _piecewise_vector_valid(result, centroids, labels), "piecewise vector must rebuild centroid rows selected by labels and then flatten")
def bicluster_piecewise_vector(
    centroids: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Rebuild one piecewise vector from supplied k-means centroids and labels."""
    centroid_values = np.asarray(centroids, dtype=np.float64)
    label_values = np.asarray(labels, dtype=np.int64)
    return np.asarray(centroid_values[label_values].ravel(), dtype=np.float64)


@register_atom(witness_bicluster_piecewise_residual_norms)
@icontract.require(lambda vectors, piecewise_vectors: _residual_inputs_valid(vectors, piecewise_vectors), "vectors and piecewise_vectors must be finite nonempty matrices with the same shape")
@icontract.ensure(lambda result, vectors: _residual_norms_valid(result, vectors), "residual norms must be a finite nonnegative vector with one value per row")
def bicluster_piecewise_residual_norms(
    vectors: NDArray[np.float64],
    piecewise_vectors: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute rowwise residual norms between vectors and supplied piecewise approximations."""
    vector_values = np.asarray(vectors, dtype=np.float64)
    piecewise_values = np.asarray(piecewise_vectors, dtype=np.float64)
    return np.asarray(np.linalg.norm(vector_values - piecewise_values, axis=1), dtype=np.float64)


@register_atom(witness_bicluster_select_best_piecewise_vectors)
@icontract.require(lambda vectors, residual_norms, n_best: _selection_inputs_valid(vectors, residual_norms, n_best), "vectors must be finite, residual_norms must align with their rows, and n_best must be between one and the number of vectors")
@icontract.ensure(lambda result, vectors, residual_norms, n_best: _selected_vectors_valid(result, vectors, residual_norms, n_best), "selected vectors must be the rows with the smallest residual norms")
def bicluster_select_best_piecewise_vectors(
    vectors: NDArray[np.float64],
    residual_norms: NDArray[np.float64],
    n_best: int,
) -> NDArray[np.float64]:
    """Select the vectors with the smallest piecewise residual norms."""
    vector_values = np.asarray(vectors, dtype=np.float64)
    order = np.argsort(np.asarray(residual_norms, dtype=np.float64))[: int(n_best)]
    return np.asarray(vector_values[order], dtype=np.float64)


@register_atom(witness_bicluster_project_dense)
@icontract.require(lambda data, vectors: _project_inputs_valid(data, vectors), "data and vectors must be finite dense matrices with aligned inner dimensions")
@icontract.ensure(lambda result, data, vectors: _projected_result_valid(result, data, vectors), "projected data must be finite with shape (n_samples, n_projected_components)")
def bicluster_project_dense(
    data: NDArray[np.float64],
    vectors: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Project dense biclustering data onto supplied vectors before clustering."""
    return np.asarray(np.asarray(data, dtype=np.float64) @ np.asarray(vectors, dtype=np.float64), dtype=np.float64)
