"""HDBSCAN DBSCAN-clustering helper atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_hdbscan_dbscan_infinite_mask,
    witness_hdbscan_dbscan_labels,
    witness_hdbscan_dbscan_missing_mask,
)


_HDBSCAN_INFINITE_LABEL = -2
_HDBSCAN_MISSING_LABEL = -3


def _integer_vector(value: object) -> bool:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.issubdtype(array.dtype, np.integer))


def _same_shape(values_a: object, values_b: object) -> bool:
    return np.asarray(values_a).shape == np.asarray(values_b).shape


def _boolean_vector_like(result: object, labels: object) -> bool:
    values = np.asarray(result)
    source = np.asarray(labels)
    return bool(values.shape == source.shape and values.dtype == np.bool_)


def _label_result_like(result: object, labels_at_cut: object) -> bool:
    values = np.asarray(result)
    source = np.asarray(labels_at_cut)
    return bool(values.shape == source.shape and np.issubdtype(values.dtype, np.integer))


@register_atom(witness_hdbscan_dbscan_infinite_mask)
@icontract.require(lambda fitted_labels: _integer_vector(fitted_labels), "fitted_labels must be a one-dimensional integer vector")
@icontract.ensure(lambda result, fitted_labels: _boolean_vector_like(result, fitted_labels), "result must be a boolean mask aligned with fitted_labels")
def hdbscan_dbscan_infinite_mask(
    fitted_labels: NDArray[np.int_],
) -> NDArray[np.bool_]:
    """Infer HDBSCAN's infinite-row mask from labels produced during fit."""
    return np.asarray(np.asarray(fitted_labels) == _HDBSCAN_INFINITE_LABEL, dtype=np.bool_)


@register_atom(witness_hdbscan_dbscan_missing_mask)
@icontract.require(lambda fitted_labels: _integer_vector(fitted_labels), "fitted_labels must be a one-dimensional integer vector")
@icontract.ensure(lambda result, fitted_labels: _boolean_vector_like(result, fitted_labels), "result must be a boolean mask aligned with fitted_labels")
def hdbscan_dbscan_missing_mask(
    fitted_labels: NDArray[np.int_],
) -> NDArray[np.bool_]:
    """Infer HDBSCAN's missing-row mask from labels produced during fit."""
    return np.asarray(np.asarray(fitted_labels) == _HDBSCAN_MISSING_LABEL, dtype=np.bool_)


@register_atom(witness_hdbscan_dbscan_labels)
@icontract.require(lambda labels_at_cut: _integer_vector(labels_at_cut), "labels_at_cut must be a one-dimensional integer vector")
@icontract.require(lambda labels_at_cut, infinite_mask: _same_shape(labels_at_cut, infinite_mask), "infinite_mask must align with labels_at_cut")
@icontract.require(lambda labels_at_cut, missing_mask: _same_shape(labels_at_cut, missing_mask), "missing_mask must align with labels_at_cut")
@icontract.require(lambda infinite_mask: np.asarray(infinite_mask).dtype == np.bool_, "infinite_mask must be boolean")
@icontract.require(lambda missing_mask: np.asarray(missing_mask).dtype == np.bool_, "missing_mask must be boolean")
@icontract.ensure(lambda result, labels_at_cut: _label_result_like(result, labels_at_cut), "result must be an integer label vector aligned with labels_at_cut")
def hdbscan_dbscan_labels(
    labels_at_cut: NDArray[np.int_],
    infinite_mask: NDArray[np.bool_],
    missing_mask: NDArray[np.bool_],
) -> NDArray[np.int_]:
    """Apply HDBSCAN's infinite and missing outlier overrides after labelling at cut."""
    labels = np.asarray(labels_at_cut).copy()
    labels[np.asarray(infinite_mask, dtype=np.bool_)] = _HDBSCAN_INFINITE_LABEL
    labels[np.asarray(missing_mask, dtype=np.bool_)] = _HDBSCAN_MISSING_LABEL
    return labels
