"""Sklearn coordinate-descent path-residual split slicing atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from scipy import sparse

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_path_residuals_X_test_slice,
    witness_cd_path_residuals_X_train_slice,
    witness_cd_path_residuals_y_test_slice,
    witness_cd_path_residuals_y_train_slice,
)


def _sliceable(value: object) -> bool:
    return hasattr(value, "__getitem__")


def _valid_index(value: object) -> bool:
    return isinstance(value, slice) or hasattr(value, "__iter__")


def _sparse_equal(left: object, right: object) -> bool:
    if not (sparse.issparse(left) and sparse.issparse(right)):
        return False
    return bool(left.shape == right.shape and (left != right).nnz == 0)


def _slice_equal(result: object, source: object, index: object) -> bool:
    try:
        expected = source[index]  # type: ignore[index]
    except (IndexError, TypeError, ValueError):
        return False
    if sparse.issparse(result) or sparse.issparse(expected):
        return _sparse_equal(result, expected)
    return bool(np.array_equal(np.asarray(result), np.asarray(expected)))


@register_atom(witness_cd_path_residuals_X_train_slice)
@icontract.require(lambda X: _sliceable(X), "X must support indexing")
@icontract.require(lambda train: _valid_index(train), "train must be a slice or iterable index")
@icontract.ensure(
    lambda result, X, train: _slice_equal(result, X, train),
    "X_train must equal X[train]",
)
def cd_path_residuals_X_train_slice(X: object, train: object) -> object:
    """Return the training feature slice used by _path_residuals."""
    return X[train]  # type: ignore[index]


@register_atom(witness_cd_path_residuals_y_train_slice)
@icontract.require(lambda y: _sliceable(y), "y must support indexing")
@icontract.require(lambda train: _valid_index(train), "train must be a slice or iterable index")
@icontract.ensure(
    lambda result, y, train: _slice_equal(result, y, train),
    "y_train must equal y[train]",
)
def cd_path_residuals_y_train_slice(y: object, train: object) -> object:
    """Return the training target slice used by _path_residuals."""
    return y[train]  # type: ignore[index]


@register_atom(witness_cd_path_residuals_X_test_slice)
@icontract.require(lambda X: _sliceable(X), "X must support indexing")
@icontract.require(lambda test: _valid_index(test), "test must be a slice or iterable index")
@icontract.ensure(
    lambda result, X, test: _slice_equal(result, X, test),
    "X_test must equal X[test]",
)
def cd_path_residuals_X_test_slice(X: object, test: object) -> object:
    """Return the test feature slice used by _path_residuals."""
    return X[test]  # type: ignore[index]


@register_atom(witness_cd_path_residuals_y_test_slice)
@icontract.require(lambda y: _sliceable(y), "y must support indexing")
@icontract.require(lambda test: _valid_index(test), "test must be a slice or iterable index")
@icontract.ensure(
    lambda result, y, test: _slice_equal(result, y, test),
    "y_test must equal y[test]",
)
def cd_path_residuals_y_test_slice(y: object, test: object) -> object:
    """Return the test target slice used by _path_residuals."""
    return y[test]  # type: ignore[index]
