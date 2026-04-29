"""Multioutput chain fit-step data helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_chain_fit_dense_step_features,
    witness_chain_fit_sparse_step_features,
    witness_chain_fit_step_feature_limit,
    witness_chain_fit_target_column,
)


def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _nonnegative_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 0)


def _order_valid(order: NDArray[np.int64], n_outputs: int) -> bool:
    values = np.asarray(order)
    return bool(
        values.ndim == 1
        and values.shape == (n_outputs,)
        and np.issubdtype(values.dtype, np.integer)
        and np.array_equal(np.sort(values), np.arange(n_outputs, dtype=np.int64))
    )


def _finite_target_matrix(Y: NDArray[np.float64]) -> bool:
    values = np.asarray(Y, dtype=np.float64)
    return bool(values.ndim == 2 and values.shape[0] >= 1 and values.shape[1] >= 1 and np.all(np.isfinite(values)))


def _target_column_valid(result: NDArray[np.float64], Y: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    y_values = np.asarray(Y, dtype=np.float64)
    return bool(values.ndim == 1 and values.shape == (y_values.shape[0],) and np.all(np.isfinite(values)))


def _dense_augmented_valid(X_aug: NDArray[np.float64], feature_limit: int) -> bool:
    values = np.asarray(X_aug, dtype=np.float64)
    return bool(
        values.ndim == 2
        and values.shape[0] >= 1
        and values.shape[1] >= 1
        and np.all(np.isfinite(values))
        and 1 <= feature_limit <= values.shape[1]
    )


def _dense_prefix_valid(result: NDArray[np.float64], X_aug: NDArray[np.float64], feature_limit: int) -> bool:
    values = np.asarray(result, dtype=np.float64)
    x_values = np.asarray(X_aug, dtype=np.float64)
    return bool(values.shape == (x_values.shape[0], feature_limit) and np.allclose(values, x_values[:, :feature_limit]))


def _sparse_augmented_valid(X_aug: object, feature_limit: int) -> bool:
    return bool(sp.issparse(X_aug) and X_aug.shape[0] >= 1 and X_aug.shape[1] >= 1 and 1 <= feature_limit <= X_aug.shape[1])


def _sparse_prefix_valid(result: object, X_aug: object, feature_limit: int) -> bool:
    return bool(
        sp.issparse(result)
        and sp.issparse(X_aug)
        and result.shape == (X_aug.shape[0], feature_limit)
        and np.allclose(result.toarray(), X_aug[:, :feature_limit].toarray())
    )


@register_atom(witness_chain_fit_step_feature_limit)
@icontract.require(lambda n_base_features: _positive_int(n_base_features), "n_base_features must be a positive integer")
@icontract.require(lambda chain_idx: _nonnegative_int(chain_idx), "chain_idx must be a nonnegative integer")
@icontract.ensure(lambda result, n_base_features, chain_idx: isinstance(result, int) and result == (n_base_features + chain_idx), "feature limit must equal n_base_features + chain_idx")
def chain_fit_step_feature_limit(n_base_features: int, chain_idx: int) -> int:
    """Return sklearn's feature-prefix width for one chain fit step."""
    return int(n_base_features + chain_idx)


@register_atom(witness_chain_fit_target_column)
@icontract.require(lambda Y: _finite_target_matrix(Y), "Y must be a finite nonempty 2D target matrix")
@icontract.require(lambda Y, order: _order_valid(order, np.asarray(Y).shape[1]), "order must be a full permutation of the output columns")
@icontract.require(lambda Y, chain_idx: _nonnegative_int(chain_idx) and chain_idx < np.asarray(Y).shape[1], "chain_idx must select an existing output")
@icontract.ensure(lambda result, Y: _target_column_valid(result, Y), "target column must be a finite 1D vector with one value per sample")
def chain_fit_target_column(
    Y: NDArray[np.float64],
    order: NDArray[np.int64],
    chain_idx: int,
) -> NDArray[np.float64]:
    """Select one ordered target column for a chain fit step."""
    y_values = np.asarray(Y, dtype=np.float64)
    order_values = np.asarray(order, dtype=np.int64)
    return np.asarray(y_values[:, order_values[chain_idx]], dtype=np.float64)


@register_atom(witness_chain_fit_dense_step_features)
@icontract.require(lambda X_aug, feature_limit: _dense_augmented_valid(X_aug, feature_limit), "X_aug must be a finite dense augmented matrix and feature_limit a valid prefix width")
@icontract.ensure(lambda result, X_aug, feature_limit: _dense_prefix_valid(result, X_aug, feature_limit), "dense step features must equal the selected feature prefix")
def chain_fit_dense_step_features(
    X_aug: NDArray[np.float64],
    feature_limit: int,
) -> NDArray[np.float64]:
    """Slice the dense augmented chain design matrix to sklearn's fit-time prefix."""
    values = np.asarray(X_aug, dtype=np.float64)
    return np.asarray(values[:, :feature_limit], dtype=np.float64)


@register_atom(witness_chain_fit_sparse_step_features)
@icontract.require(lambda X_aug, feature_limit: _sparse_augmented_valid(X_aug, feature_limit), "X_aug must be a sparse augmented matrix and feature_limit a valid prefix width")
@icontract.ensure(lambda result, X_aug, feature_limit: _sparse_prefix_valid(result, X_aug, feature_limit), "sparse step features must equal the selected feature prefix")
def chain_fit_sparse_step_features(
    X_aug: sp.spmatrix | sp.sparray,
    feature_limit: int,
) -> sp.spmatrix | sp.sparray:
    """Slice the sparse augmented chain design matrix to sklearn's fit-time prefix."""
    return X_aug[:, :feature_limit]
