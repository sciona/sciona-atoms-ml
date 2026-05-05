"""Partial-dependence categorical-preflight shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_partial_dependence_categorical_bool_size_guard_required,
    witness_partial_dependence_categorical_bool_size_message,
    witness_partial_dependence_categorical_dtype_message,
    witness_partial_dependence_categorical_dtype_supported,
    witness_partial_dependence_categorical_empty_guard_required,
    witness_partial_dependence_categorical_empty_message,
)


def _nonnegative_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 0)


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _dtype_kind(value: object) -> bool:
    return isinstance(value, str) and len(value) >= 1


def _dtype_name(value: object) -> bool:
    return isinstance(value, str) and len(value) >= 1


@register_atom(witness_partial_dependence_categorical_empty_guard_required)
@icontract.require(lambda size: _nonnegative_int(size), "size must be a nonnegative integer")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def partial_dependence_categorical_empty_guard_required(
    size: int,
) -> bool:
    """Decide whether partial_dependence rejects an empty categorical_features input."""
    return int(size) == 0


@register_atom(witness_partial_dependence_categorical_empty_message)
@icontract.require(lambda size: _nonnegative_int(size), "size must be a nonnegative integer")
@icontract.ensure(lambda result: isinstance(result, str) and len(result) >= 1, "result must be a nonempty message")
def partial_dependence_categorical_empty_message(
    size: int,
) -> str:
    """Format sklearn's empty categorical_features ValueError message."""
    del size
    return (
        "Passing an empty list (`[]`) to `categorical_features` is not "
        "supported. Use `None` instead to indicate that there are no "
        "categorical features."
    )


@register_atom(witness_partial_dependence_categorical_bool_size_guard_required)
@icontract.require(lambda size: _nonnegative_int(size), "size must be a nonnegative integer")
@icontract.require(lambda n_features: _positive_int(n_features), "n_features must be a positive integer")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def partial_dependence_categorical_bool_size_guard_required(
    size: int,
    n_features: int,
) -> bool:
    """Decide whether partial_dependence rejects a boolean categorical mask of the wrong length."""
    return int(size) != int(n_features)


@register_atom(witness_partial_dependence_categorical_bool_size_message)
@icontract.require(lambda size: _nonnegative_int(size), "size must be a nonnegative integer")
@icontract.require(lambda n_features: _positive_int(n_features), "n_features must be a positive integer")
@icontract.ensure(lambda result: isinstance(result, str) and len(result) >= 1, "result must be a nonempty message")
def partial_dependence_categorical_bool_size_message(
    size: int,
    n_features: int,
) -> str:
    """Format sklearn's boolean categorical_features length mismatch message."""
    return (
        "When `categorical_features` is a boolean array-like, "
        "the array should be of shape (n_features,). Got "
        f"{int(size)} elements while `X` contains "
        f"{int(n_features)} features."
    )


@register_atom(witness_partial_dependence_categorical_dtype_supported)
@icontract.require(lambda dtype_kind: _dtype_kind(dtype_kind), "dtype_kind must be a nonempty dtype-kind string")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def partial_dependence_categorical_dtype_supported(
    dtype_kind: str,
) -> bool:
    """Decide whether partial_dependence accepts the categorical_features dtype kind."""
    return dtype_kind in ("b", "i", "O", "U")


@register_atom(witness_partial_dependence_categorical_dtype_message)
@icontract.require(lambda dtype_name: _dtype_name(dtype_name), "dtype_name must be a nonempty dtype string")
@icontract.ensure(lambda result: isinstance(result, str) and len(result) >= 1, "result must be a nonempty message")
def partial_dependence_categorical_dtype_message(
    dtype_name: str,
) -> str:
    """Format sklearn's unsupported categorical_features dtype message."""
    return (
        "Expected `categorical_features` to be an array-like of boolean,"
        f" integer, or string. Got {dtype_name} instead."
    )
