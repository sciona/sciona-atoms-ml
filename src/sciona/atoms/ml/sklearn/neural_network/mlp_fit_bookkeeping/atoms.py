"""MLP fit-shell bookkeeping helper atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Sequence

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_mlp_batch_size,
    witness_mlp_batch_size_warning_required,
    witness_mlp_first_pass_required,
    witness_mlp_hidden_layer_sizes,
    witness_mlp_partial_fit_require_no_early_stopping,
)


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _int_not_bool(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _hidden_sizes_input_valid(hidden_layer_sizes: object) -> bool:
    if _int_not_bool(hidden_layer_sizes):
        return True
    if isinstance(hidden_layer_sizes, Sequence) and not isinstance(hidden_layer_sizes, (str, bytes)):
        values = list(hidden_layer_sizes)
        return bool(len(values) >= 1 and all(_int_not_bool(value) for value in values))
    return False


def _hidden_sizes_result_valid(result: tuple[int, ...]) -> bool:
    return bool(isinstance(result, tuple) and len(result) >= 1 and all(_positive_int(value) for value in result))


def _bool_valid(value: object) -> bool:
    return isinstance(value, bool)


def _batch_size_valid(value: object) -> bool:
    return value == "auto" or _positive_int(value)


@register_atom(witness_mlp_hidden_layer_sizes)
@icontract.require(lambda hidden_layer_sizes: _hidden_sizes_input_valid(hidden_layer_sizes), "hidden_layer_sizes must be an integer or a nonempty sequence of integers")
@icontract.ensure(lambda result: _hidden_sizes_result_valid(result), "hidden-layer sizes must normalize to a nonempty tuple of positive integers")
def mlp_hidden_layer_sizes(
    hidden_layer_sizes: int | Sequence[int],
) -> tuple[int, ...]:
    """Normalize MLP hidden-layer sizes the way sklearn's fit shell does before validation."""
    if _int_not_bool(hidden_layer_sizes):
        if int(hidden_layer_sizes) <= 0:
            raise ValueError(f"hidden_layer_sizes must be > 0, got {[int(hidden_layer_sizes)]}.")
        return (int(hidden_layer_sizes),)
    values = tuple(int(value) for value in hidden_layer_sizes)
    if np.any(np.asarray(values, dtype=np.int64) <= 0):
        raise ValueError(f"hidden_layer_sizes must be > 0, got {list(values)}.")
    return values


@register_atom(witness_mlp_first_pass_required)
@icontract.require(lambda has_coefs: _bool_valid(has_coefs), "has_coefs must be boolean")
@icontract.require(lambda warm_start: _bool_valid(warm_start), "warm_start must be boolean")
@icontract.require(lambda incremental: _bool_valid(incremental), "incremental must be boolean")
@icontract.ensure(lambda result: _bool_valid(result), "first-pass flag must be boolean")
def mlp_first_pass_required(
    *,
    has_coefs: bool,
    warm_start: bool,
    incremental: bool,
) -> bool:
    """Resolve sklearn's first-pass condition for MLP fit and partial_fit."""
    return (not has_coefs) or (not warm_start and not incremental)


@register_atom(witness_mlp_partial_fit_require_no_early_stopping)
@icontract.require(lambda early_stopping: _bool_valid(early_stopping), "early_stopping must be boolean")
@icontract.require(lambda incremental: _bool_valid(incremental), "incremental must be boolean")
@icontract.ensure(lambda result: _bool_valid(result), "guard result must be boolean")
def mlp_partial_fit_require_no_early_stopping(
    *,
    early_stopping: bool,
    incremental: bool,
) -> bool:
    """Enforce sklearn's partial_fit restriction against early stopping."""
    if early_stopping and incremental:
        raise ValueError("partial_fit does not support early_stopping=True")
    return True


@register_atom(witness_mlp_batch_size_warning_required)
@icontract.require(lambda batch_size: _batch_size_valid(batch_size), "batch_size must be 'auto' or a positive integer")
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be a positive integer")
@icontract.ensure(lambda result: _bool_valid(result), "warning flag must be boolean")
def mlp_batch_size_warning_required(
    batch_size: int | str,
    *,
    n_samples: int,
) -> bool:
    """Return whether sklearn's stochastic MLP fit would emit the batch-size clipping warning."""
    return batch_size != "auto" and int(batch_size) > n_samples


@register_atom(witness_mlp_batch_size)
@icontract.require(lambda batch_size: _batch_size_valid(batch_size), "batch_size must be 'auto' or a positive integer")
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be a positive integer")
@icontract.ensure(lambda result: _positive_int(result), "resolved batch_size must be a positive integer")
def mlp_batch_size(
    batch_size: int | str,
    *,
    n_samples: int,
) -> int:
    """Resolve sklearn's stochastic MLP batch size after auto/default and clipping logic."""
    if batch_size == "auto":
        return int(min(200, n_samples))
    return int(np.clip(int(batch_size), 1, n_samples))
