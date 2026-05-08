"""Sklearn coordinate-descent _set_order helper atoms."""

from __future__ import annotations

import icontract
import numpy as np
from scipy import sparse

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_set_order_conversion_required,
    witness_cd_set_order_invalid_order_message,
    witness_cd_set_order_invalid_order_required,
    witness_cd_set_order_outputs,
    witness_cd_set_order_sparse_format,
)


_VALID_ORDERS = frozenset({None, "C", "F"})


def _valid_order(order: object) -> bool:
    return order in _VALID_ORDERS


def _convertible_array(value: object) -> bool:
    if sparse.issparse(value):
        return True
    try:
        np.asarray(value)
    except (TypeError, ValueError):
        return False
    return True


def _matches_ordered_value(original: object, converted: object, order: object) -> bool:
    if order is None:
        return converted is original
    sparse_format = "csc" if order == "F" else "csr"
    if sparse.issparse(original):
        return (
            sparse.issparse(converted)
            and converted.getformat() == sparse_format
            and np.array_equal(converted.toarray(), original.toarray())
        )
    converted_array = np.asarray(converted)
    original_array = np.asarray(original)
    if not np.array_equal(converted_array, original_array):
        return False
    if order == "F":
        return bool(converted_array.flags["F_CONTIGUOUS"])
    return bool(converted_array.flags["C_CONTIGUOUS"])


@register_atom(witness_cd_set_order_invalid_order_required)
@icontract.ensure(
    lambda result, order: isinstance(result, bool) and result == (order not in _VALID_ORDERS),
    "invalid order guard must match order not in [None, 'C', 'F']",
)
def cd_set_order_invalid_order_required(order: object) -> bool:
    """Return whether _set_order should raise for an invalid order."""
    return order not in _VALID_ORDERS


@register_atom(witness_cd_set_order_invalid_order_message)
@icontract.ensure(
    lambda result, order: isinstance(result, str)
    and result == "Unknown value for order. Got {} instead of None, 'C' or 'F'.".format(order),
    "invalid order message must match sklearn formatting",
)
def cd_set_order_invalid_order_message(order: object) -> str:
    """Return the invalid-order ValueError text used by _set_order."""
    return "Unknown value for order. Got {} instead of None, 'C' or 'F'.".format(order)


@register_atom(witness_cd_set_order_conversion_required)
@icontract.require(lambda order: _valid_order(order), "order must be None, 'C', or 'F'")
@icontract.ensure(
    lambda result, order: isinstance(result, bool) and result == (order is not None),
    "conversion predicate must match order is not None",
)
def cd_set_order_conversion_required(order: str | None) -> bool:
    """Return whether _set_order converts X and y."""
    return order is not None


@register_atom(witness_cd_set_order_sparse_format)
@icontract.require(lambda order: order in {"C", "F"}, "order must be 'C' or 'F'")
@icontract.ensure(
    lambda result, order: result == ("csc" if order == "F" else "csr"),
    "sparse format must be csc for F order and csr otherwise",
)
def cd_set_order_sparse_format(order: str) -> str:
    """Return the sparse format selected by _set_order."""
    return "csc" if order == "F" else "csr"


@register_atom(witness_cd_set_order_outputs)
@icontract.require(lambda order: _valid_order(order), "order must be None, 'C', or 'F'")
@icontract.require(lambda X: _convertible_array(X), "X must be array-like or sparse")
@icontract.require(lambda y: _convertible_array(y), "y must be array-like or sparse")
@icontract.ensure(
    lambda result, X, y, order: isinstance(result, tuple)
    and len(result) == 2
    and _matches_ordered_value(X, result[0], order)
    and _matches_ordered_value(y, result[1], order),
    "_set_order outputs must preserve values and apply requested dense or sparse order",
)
def cd_set_order_outputs(X: object, y: object, order: str | None = "C") -> tuple[object, object]:
    """Return the X and y outputs produced by _set_order."""
    if order is None:
        return X, y

    sparse_format = "csc" if order == "F" else "csr"
    if sparse.issparse(X):
        ordered_X = X.asformat(sparse_format, copy=False)
    else:
        ordered_X = np.asarray(X, order=order)

    if sparse.issparse(y):
        ordered_y = y.asformat(sparse_format)
    else:
        ordered_y = np.asarray(y, order=order)

    return ordered_X, ordered_y
