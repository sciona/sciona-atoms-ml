"""Ghost witnesses for sklearn coordinate-descent _set_order atoms."""

from __future__ import annotations


def witness_cd_set_order_invalid_order_required(order: object) -> object:
    """Describe the invalid order guard in _set_order."""
    return order


def witness_cd_set_order_invalid_order_message(order: object) -> object:
    """Describe the invalid order ValueError message in _set_order."""
    return order


def witness_cd_set_order_conversion_required(order: object) -> object:
    """Describe whether _set_order converts X and y."""
    return order


def witness_cd_set_order_sparse_format(order: object) -> object:
    """Describe the sparse format selected by _set_order."""
    return order


def witness_cd_set_order_outputs(X: object, y: object, order: object) -> object:
    """Describe the X and y outputs returned by _set_order."""
    return X, y, order
