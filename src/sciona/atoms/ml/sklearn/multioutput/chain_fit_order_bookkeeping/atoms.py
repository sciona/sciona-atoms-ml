"""Multioutput chain fit-order bookkeeping helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_chain_fit_log_message,
    witness_chain_fit_require_valid_order,
    witness_chain_fit_tuple_order_array,
)


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _int_tuple(value: object) -> bool:
    return isinstance(value, tuple) and len(value) >= 1 and all(isinstance(item, int) and not isinstance(item, bool) for item in value)


def _order_like_valid(value: object) -> bool:
    if isinstance(value, list):
        return len(value) >= 1 and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    array = np.asarray(value)
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.issubdtype(array.dtype, np.integer))


def _permutation_valid(value: NDArray[np.int64], n_outputs: int) -> bool:
    order_values = np.asarray(value)
    return bool(
        order_values.ndim == 1
        and order_values.shape[0] == n_outputs
        and np.issubdtype(order_values.dtype, np.integer)
        and sorted(int(item) for item in order_values) == list(range(n_outputs))
    )


def _message_valid(value: object) -> bool:
    return isinstance(value, str) and len(value) >= 1


@register_atom(witness_chain_fit_tuple_order_array)
@icontract.require(lambda order: _int_tuple(order), "order must be a nonempty tuple of integers")
@icontract.ensure(lambda result, order: np.array_equal(result, np.asarray(order, dtype=np.int64)), "result must preserve tuple order values")
def chain_fit_tuple_order_array(order: tuple[int, ...]) -> NDArray[np.int64]:
    """Convert tuple-configured chain order into sklearn's ndarray form."""
    return np.asarray(order, dtype=np.int64)


@register_atom(witness_chain_fit_require_valid_order)
@icontract.require(lambda order, n_outputs: _order_like_valid(order) and _positive_int(n_outputs), "order must be a nonempty 1D integer sequence and n_outputs must be positive")
@icontract.ensure(lambda result: result is True, "result must be True when validation succeeds")
def chain_fit_require_valid_order(
    order: list[int] | NDArray[np.int64],
    n_outputs: int,
) -> bool:
    """Require that an explicit chain order is a full output permutation."""
    order_values = np.asarray(order, dtype=np.int64)
    if not _permutation_valid(order_values, n_outputs):
        raise ValueError("invalid order")
    return True


@register_atom(witness_chain_fit_log_message)
@icontract.require(lambda verbose: isinstance(verbose, bool), "verbose must be boolean")
@icontract.require(lambda estimator_idx: _positive_int(estimator_idx), "estimator_idx must be positive")
@icontract.require(lambda n_estimators: _positive_int(n_estimators), "n_estimators must be positive")
@icontract.require(lambda estimator_idx, n_estimators: estimator_idx <= n_estimators, "estimator_idx must not exceed n_estimators")
@icontract.require(lambda processing_msg: _message_valid(processing_msg), "processing_msg must be a nonempty string")
@icontract.ensure(lambda result: result is None or _message_valid(result), "result must be None or a nonempty string")
def chain_fit_log_message(
    verbose: bool,
    *,
    estimator_idx: int,
    n_estimators: int,
    processing_msg: str,
) -> str | None:
    """Format sklearn's verbose chain fit progress message."""
    if not verbose:
        return None
    return f"({estimator_idx} of {n_estimators}) {processing_msg}"
