"""Multioutput chain prediction bookkeeping helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_chain_prediction_feature_buffer,
    witness_chain_prediction_method_name,
    witness_chain_prediction_previous_predictions,
    witness_chain_prediction_output_buffer,
    witness_chain_sparse_hstack_base,
)


def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _nonempty_method_name(value: str | None) -> bool:
    return value is None or (isinstance(value, str) and len(value) >= 1)


def _finite_dense_2d(values: NDArray[np.float64]) -> bool:
    array = np.asarray(values, dtype=np.float64)
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _finite_dense_2d_allow_empty_columns(values: NDArray[np.float64]) -> bool:
    array = np.asarray(values, dtype=np.float64)
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 0 and np.all(np.isfinite(array)))


def _buffer_valid(result: NDArray[np.float64], n_samples: int, n_outputs: int) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(
        values.shape == (n_samples, n_outputs)
        and np.all(np.isfinite(values))
        and np.allclose(values, 0.0)
    )


def _prefix_inputs_valid(feature_chain: NDArray[np.float64], chain_idx: int) -> bool:
    values = np.asarray(feature_chain, dtype=np.float64)
    return bool(_finite_dense_2d(feature_chain) and 0 <= chain_idx <= values.shape[1])


def _prefix_valid(result: NDArray[np.float64], feature_chain: NDArray[np.float64], chain_idx: int) -> bool:
    values = np.asarray(result, dtype=np.float64)
    chain_values = np.asarray(feature_chain, dtype=np.float64)
    return bool(
        _finite_dense_2d_allow_empty_columns(result)
        and values.shape == (chain_values.shape[0], chain_idx)
        and np.array_equal(values, chain_values[:, :chain_idx])
    )


def _sparse_2d(values: sp.spmatrix | sp.sparray) -> bool:
    return bool(sp.issparse(values) and len(values.shape) == 2 and values.shape[0] >= 1 and values.shape[1] >= 1)


def _same_sparse_contents(left: sp.spmatrix | sp.sparray, right: sp.spmatrix | sp.sparray) -> bool:
    return bool(
        _sparse_2d(left)
        and _sparse_2d(right)
        and left.shape == right.shape
        and np.array_equal(left.toarray(), right.toarray())
    )


@register_atom(witness_chain_prediction_method_name)
@icontract.require(
    lambda chain_method_name: _nonempty_method_name(chain_method_name),
    "chain_method_name must be None or a nonempty string",
)
@icontract.ensure(
    lambda result: isinstance(result, str) and len(result) >= 1,
    "resolved chain method name must be a nonempty string",
)
def chain_prediction_method_name(chain_method_name: str | None = None) -> str:
    """Resolve sklearn's prediction-time chain method fallback."""
    if chain_method_name is None:
        return "predict"
    return chain_method_name


@register_atom(witness_chain_prediction_output_buffer)
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be a positive integer")
@icontract.require(lambda n_outputs: _positive_int(n_outputs), "n_outputs must be a positive integer")
@icontract.ensure(
    lambda result, n_samples, n_outputs: _buffer_valid(result, n_samples, n_outputs),
    "output buffer must be an all-zero finite sample-by-output matrix",
)
def chain_prediction_output_buffer(n_samples: int, n_outputs: int) -> NDArray[np.float64]:
    """Allocate sklearn's zero-initialized output-prediction chain buffer."""
    return np.zeros((n_samples, n_outputs), dtype=np.float64)


@register_atom(witness_chain_prediction_feature_buffer)
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be a positive integer")
@icontract.require(lambda n_outputs: _positive_int(n_outputs), "n_outputs must be a positive integer")
@icontract.ensure(
    lambda result, n_samples, n_outputs: _buffer_valid(result, n_samples, n_outputs),
    "feature buffer must be an all-zero finite sample-by-output matrix",
)
def chain_prediction_feature_buffer(n_samples: int, n_outputs: int) -> NDArray[np.float64]:
    """Allocate sklearn's zero-initialized feature-prediction chain buffer."""
    return np.zeros((n_samples, n_outputs), dtype=np.float64)


@register_atom(witness_chain_prediction_previous_predictions)
@icontract.require(
    lambda feature_chain, chain_idx: _prefix_inputs_valid(feature_chain, chain_idx),
    "feature_chain must be finite and chain_idx must select a valid prefix width",
)
@icontract.ensure(
    lambda result, feature_chain, chain_idx: _prefix_valid(result, feature_chain, chain_idx),
    "previous predictions must be the feature buffer prefix through chain_idx",
)
def chain_prediction_previous_predictions(
    feature_chain: NDArray[np.float64],
    chain_idx: int,
) -> NDArray[np.float64]:
    """Slice sklearn's previously filled chain-prediction columns for one step."""
    return np.asarray(np.asarray(feature_chain, dtype=np.float64)[:, :chain_idx], dtype=np.float64)


@register_atom(witness_chain_sparse_hstack_base)
@icontract.require(lambda X: _sparse_2d(X), "X must be a nonempty 2D sparse matrix or sparse array")
@icontract.ensure(
    lambda result, X: _same_sparse_contents(result, X),
    "prepared sparse input must preserve shape and numeric contents",
)
def chain_sparse_hstack_base(X: sp.spmatrix | sp.sparray) -> sp.spmatrix | sp.sparray:
    """Normalize sklearn's sparse DOK array input before prediction-time hstack."""
    if not sp.isspmatrix(X) and getattr(X, "format", None) == "dok":
        return sp.coo_array(X)
    return X
