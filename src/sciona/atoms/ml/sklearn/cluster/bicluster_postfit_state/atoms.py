"""Spectral biclustering and coclustering post-fit state atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_bicluster_fit_column_labels,
    witness_bicluster_fit_columns,
    witness_bicluster_fit_return_self,
    witness_bicluster_fit_row_labels,
    witness_bicluster_fit_rows,
)


def _nonnegative_int_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.int64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(array >= 0))


def _bool_matrix(values: object) -> bool:
    array = np.asarray(values)
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and array.dtype == np.bool_)


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and value != ""


@register_atom(witness_bicluster_fit_row_labels)
@icontract.require(lambda row_labels: _nonnegative_int_vector(row_labels), "row_labels must be a nonempty nonnegative integer vector")
@icontract.ensure(
    lambda result, row_labels: _nonnegative_int_vector(result) and np.asarray(result).shape == np.asarray(row_labels).shape,
    "row_labels must preserve the fitted row-label shape",
)
def bicluster_fit_row_labels(
    row_labels: NDArray[np.int64],
) -> NDArray[np.int64]:
    """Expose fitted row labels from spectral biclustering or coclustering."""
    return np.asarray(row_labels, dtype=np.int64)


@register_atom(witness_bicluster_fit_column_labels)
@icontract.require(lambda column_labels: _nonnegative_int_vector(column_labels), "column_labels must be a nonempty nonnegative integer vector")
@icontract.ensure(
    lambda result, column_labels: _nonnegative_int_vector(result) and np.asarray(result).shape == np.asarray(column_labels).shape,
    "column_labels must preserve the fitted column-label shape",
)
def bicluster_fit_column_labels(
    column_labels: NDArray[np.int64],
) -> NDArray[np.int64]:
    """Expose fitted column labels from spectral biclustering or coclustering."""
    return np.asarray(column_labels, dtype=np.int64)


@register_atom(witness_bicluster_fit_rows)
@icontract.require(lambda rows: _bool_matrix(rows), "rows must be a nonempty boolean matrix")
@icontract.ensure(
    lambda result, rows: _bool_matrix(result) and np.asarray(result).shape == np.asarray(rows).shape,
    "row indicator matrix must preserve the fitted shape",
)
def bicluster_fit_rows(
    rows: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Expose the fitted row-indicator matrix from spectral biclustering or coclustering."""
    return np.asarray(rows, dtype=np.bool_)


@register_atom(witness_bicluster_fit_columns)
@icontract.require(lambda columns: _bool_matrix(columns), "columns must be a nonempty boolean matrix")
@icontract.ensure(
    lambda result, columns: _bool_matrix(result) and np.asarray(result).shape == np.asarray(columns).shape,
    "column indicator matrix must preserve the fitted shape",
)
def bicluster_fit_columns(
    columns: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Expose the fitted column-indicator matrix from spectral biclustering or coclustering."""
    return np.asarray(columns, dtype=np.bool_)


@register_atom(witness_bicluster_fit_return_self)
@icontract.require(lambda estimator_token: _nonempty_string(estimator_token), "estimator_token must be a nonempty string")
@icontract.ensure(lambda result, estimator_token: result == estimator_token, "fit return value must preserve the estimator token")
def bicluster_fit_return_self(estimator_token: str) -> str:
    """Model SpectralBiclustering.fit and SpectralCoclustering.fit returning self after state assignment."""
    return estimator_token
