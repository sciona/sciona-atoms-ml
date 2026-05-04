"""Spectral clustering label-selection atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_spectral_fit_selected_labels,
    witness_spectral_fit_use_discretize,
)


def _nonempty_string(value: object) -> bool:
    return bool(isinstance(value, str) and value != "")


def _boolean(value: object) -> bool:
    return isinstance(value, bool)


def _integer_vector(value: object) -> bool:
    array = np.asarray(value)
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.issubdtype(array.dtype, np.integer))


def _same_label_length(
    kmeans_labels: NDArray[np.int_],
    cluster_qr_labels: NDArray[np.int_],
    discretize_labels: NDArray[np.int_],
) -> bool:
    kmeans = np.asarray(kmeans_labels)
    cluster_qr = np.asarray(cluster_qr_labels)
    discretize = np.asarray(discretize_labels)
    return bool(kmeans.shape == cluster_qr.shape == discretize.shape)


def _selected_labels_valid(
    result: object,
    use_kmeans: bool,
    use_cluster_qr: bool,
    kmeans_labels: NDArray[np.int_],
    cluster_qr_labels: NDArray[np.int_],
    discretize_labels: NDArray[np.int_],
) -> bool:
    values = np.asarray(result)
    if not _integer_vector(values):
        return False
    if use_kmeans:
        expected = np.asarray(kmeans_labels)
    elif use_cluster_qr:
        expected = np.asarray(cluster_qr_labels)
    else:
        expected = np.asarray(discretize_labels)
    return bool(values.shape == expected.shape and np.array_equal(values, expected))


@register_atom(witness_spectral_fit_use_discretize)
@icontract.require(lambda assign_labels: _nonempty_string(assign_labels), "assign_labels must be a nonempty string")
@icontract.ensure(lambda result: _boolean(result), "result must be boolean")
def spectral_fit_use_discretize(assign_labels: str) -> bool:
    """Return whether SpectralClustering.fit falls through to discretize label assignment."""
    return assign_labels not in {"kmeans", "cluster_qr"}


@register_atom(witness_spectral_fit_selected_labels)
@icontract.require(lambda use_kmeans: _boolean(use_kmeans), "use_kmeans must be boolean")
@icontract.require(lambda use_cluster_qr: _boolean(use_cluster_qr), "use_cluster_qr must be boolean")
@icontract.require(lambda kmeans_labels: _integer_vector(kmeans_labels), "kmeans_labels must be a nonempty one-dimensional integer vector")
@icontract.require(lambda cluster_qr_labels: _integer_vector(cluster_qr_labels), "cluster_qr_labels must be a nonempty one-dimensional integer vector")
@icontract.require(lambda discretize_labels: _integer_vector(discretize_labels), "discretize_labels must be a nonempty one-dimensional integer vector")
@icontract.require(
    lambda kmeans_labels, cluster_qr_labels, discretize_labels: _same_label_length(
        kmeans_labels,
        cluster_qr_labels,
        discretize_labels,
    ),
    "all label vectors must share the same length",
)
@icontract.require(
    lambda use_kmeans, use_cluster_qr: not (use_kmeans and use_cluster_qr),
    "use_kmeans and use_cluster_qr cannot both be true",
)
@icontract.ensure(
    lambda result, use_kmeans, use_cluster_qr, kmeans_labels, cluster_qr_labels, discretize_labels: _selected_labels_valid(
        result,
        use_kmeans,
        use_cluster_qr,
        kmeans_labels,
        cluster_qr_labels,
        discretize_labels,
    ),
    "result must match the selected branch's labels",
)
def spectral_fit_selected_labels(
    use_kmeans: bool,
    use_cluster_qr: bool,
    kmeans_labels: NDArray[np.int_],
    cluster_qr_labels: NDArray[np.int_],
    discretize_labels: NDArray[np.int_],
) -> NDArray[np.int_]:
    """Select the final SpectralClustering label vector from supplied branch outputs."""
    if use_kmeans:
        return np.asarray(kmeans_labels).copy()
    if use_cluster_qr:
        return np.asarray(cluster_qr_labels).copy()
    return np.asarray(discretize_labels).copy()
