"""Multioutput chain augmentation helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_chain_cv_feature_column,
    witness_chain_dense_cv_feature_buffer,
    witness_chain_sparse_cv_feature_buffer,
    witness_chain_sparse_step_features,
    witness_chain_sparse_training_features,
)


def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _finite_dense_2d(values: NDArray[np.float64]) -> bool:
    array = np.asarray(values, dtype=np.float64)
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _finite_cv_result(values: NDArray[np.float64]) -> bool:
    array = np.asarray(values, dtype=np.float64)
    return bool(
        np.all(np.isfinite(array))
        and (
            (array.ndim == 1 and array.shape[0] >= 1)
            or (array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 2)
        )
    )


def _sparse_2d(values: sp.spmatrix | sp.sparray) -> bool:
    return bool(sp.issparse(values) and len(values.shape) == 2 and values.shape[0] >= 1 and values.shape[1] >= 1)


def _same_row_count_sparse_dense(
    X: sp.spmatrix | sp.sparray,
    values: NDArray[np.float64],
) -> bool:
    return bool(_sparse_2d(X) and _finite_dense_2d(values) and X.shape[0] == np.asarray(values, dtype=np.float64).shape[0])


def _dense_buffer_valid(result: NDArray[np.float64], X: NDArray[np.float64], n_outputs: int) -> bool:
    values = np.asarray(result, dtype=np.float64)
    x_values = np.asarray(X, dtype=np.float64)
    return bool(
        values.shape == (x_values.shape[0], x_values.shape[1] + n_outputs)
        and np.all(np.isfinite(values))
        and np.allclose(values[:, : x_values.shape[1]], x_values)
        and np.allclose(values[:, x_values.shape[1] :], 0.0)
    )


def _sparse_result_valid(
    result: sp.spmatrix | sp.sparray,
    n_samples: int,
    n_features: int,
) -> bool:
    return bool(sp.issparse(result) and result.shape == (n_samples, n_features))


def _cv_column_result_valid(result: NDArray[np.float64], cv_result: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    input_values = np.asarray(cv_result, dtype=np.float64)
    return bool(
        values.ndim == 1
        and values.shape == (input_values.shape[0],)
        and np.all(np.isfinite(values))
    )


def _prepare_sparse_for_hstack(values: sp.spmatrix | sp.sparray) -> sp.spmatrix | sp.sparray:
    if not sp.isspmatrix(values) and getattr(values, "format", None) == "dok":
        return sp.coo_array(values)
    return values


@register_atom(witness_chain_dense_cv_feature_buffer)
@icontract.require(lambda X: _finite_dense_2d(X), "X must be a finite dense 2D matrix")
@icontract.require(lambda n_outputs: _positive_int(n_outputs), "n_outputs must be a positive integer")
@icontract.ensure(
    lambda result, X, n_outputs: _dense_buffer_valid(result, X, n_outputs),
    "dense CV feature buffer must append one zero column per output",
)
def chain_dense_cv_feature_buffer(
    X: NDArray[np.float64],
    n_outputs: int,
) -> NDArray[np.float64]:
    """Append zero placeholder columns for dense chain fitting with cv != None."""
    x_values = np.asarray(X, dtype=np.float64)
    return np.asarray(
        np.hstack((x_values, np.zeros((x_values.shape[0], n_outputs), dtype=np.float64))),
        dtype=np.float64,
    )


@register_atom(witness_chain_sparse_cv_feature_buffer)
@icontract.require(lambda X: _sparse_2d(X), "X must be a nonempty 2D sparse matrix or sparse array")
@icontract.require(lambda n_outputs: _positive_int(n_outputs), "n_outputs must be a positive integer")
@icontract.ensure(
    lambda result, X, n_outputs: _sparse_result_valid(result, X.shape[0], X.shape[1] + n_outputs),
    "sparse CV feature buffer must append one zero column per output",
)
def chain_sparse_cv_feature_buffer(
    X: sp.spmatrix | sp.sparray,
    n_outputs: int,
) -> sp.spmatrix | sp.sparray:
    """Append sparse zero placeholder columns for chain fitting with cv != None."""
    prepared_x = _prepare_sparse_for_hstack(X)
    zeros = (
        sp.coo_matrix((prepared_x.shape[0], n_outputs))
        if sp.isspmatrix(prepared_x)
        else sp.coo_array((prepared_x.shape[0], n_outputs))
    )
    return sp.hstack((prepared_x, zeros), format="lil")


@register_atom(witness_chain_sparse_training_features)
@icontract.require(lambda X: _sparse_2d(X), "X must be a nonempty 2D sparse matrix or sparse array")
@icontract.require(lambda Y_ordered: _finite_dense_2d(Y_ordered), "Y_ordered must be a finite dense 2D matrix")
@icontract.require(
    lambda X, Y_ordered: _same_row_count_sparse_dense(X, Y_ordered),
    "X and Y_ordered must have matching sample counts",
)
@icontract.ensure(
    lambda result, X, Y_ordered: _sparse_result_valid(result, X.shape[0], X.shape[1] + np.asarray(Y_ordered).shape[1]),
    "sparse training features must append ordered target columns to the sparse design matrix",
)
def chain_sparse_training_features(
    X: sp.spmatrix | sp.sparray,
    Y_ordered: NDArray[np.float64],
) -> sp.spmatrix | sp.sparray:
    """Append ordered target columns to sparse features for cv=None chain fitting."""
    prepared_x = _prepare_sparse_for_hstack(X)
    y_values = np.asarray(Y_ordered, dtype=np.float64)
    return sp.hstack((prepared_x, y_values), format="lil").tocsr()


@register_atom(witness_chain_sparse_step_features)
@icontract.require(lambda X: _sparse_2d(X), "X must be a nonempty 2D sparse matrix or sparse array")
@icontract.require(lambda previous_predictions: _finite_dense_2d(previous_predictions), "previous_predictions must be a finite dense 2D matrix")
@icontract.require(
    lambda X, previous_predictions: _same_row_count_sparse_dense(X, previous_predictions),
    "X and previous_predictions must have matching sample counts",
)
@icontract.ensure(
    lambda result, X, previous_predictions: _sparse_result_valid(result, X.shape[0], X.shape[1] + np.asarray(previous_predictions).shape[1]),
    "sparse step features must append previous prediction columns to the sparse design matrix",
)
def chain_sparse_step_features(
    X: sp.spmatrix | sp.sparray,
    previous_predictions: NDArray[np.float64],
) -> sp.spmatrix | sp.sparray:
    """Append previous chain predictions to sparse features for prediction-time steps."""
    prepared_x = _prepare_sparse_for_hstack(X)
    previous_values = np.asarray(previous_predictions, dtype=np.float64)
    return sp.hstack((prepared_x, previous_values), format="csr")


@register_atom(witness_chain_cv_feature_column)
@icontract.require(lambda cv_result: _finite_cv_result(cv_result), "cv_result must be a finite 1D vector or a 2D matrix with at least two columns")
@icontract.ensure(
    lambda result, cv_result: _cv_column_result_valid(result, cv_result),
    "feature column must be a finite vector with one entry per sample",
)
def chain_cv_feature_column(cv_result: NDArray[np.float64]) -> NDArray[np.float64]:
    """Resolve sklearn's chain CV feature column from 1D outputs or predict_proba matrices."""
    values = np.asarray(cv_result, dtype=np.float64)
    if values.ndim == 1:
        return np.asarray(values, dtype=np.float64)
    return np.asarray(values[:, 1], dtype=np.float64)
