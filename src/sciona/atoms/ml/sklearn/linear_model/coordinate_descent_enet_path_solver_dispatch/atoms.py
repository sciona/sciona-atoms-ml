"""Sklearn coordinate-descent enet_path solver-dispatch atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_enet_path_gram_validation_required,
    witness_cd_enet_path_invalid_precompute_message,
    witness_cd_enet_path_use_dense_solver,
    witness_cd_enet_path_use_gram_solver,
    witness_cd_enet_path_use_multi_task_solver,
    witness_cd_enet_path_use_sparse_solver,
)


@register_atom(witness_cd_enet_path_gram_validation_required)
@icontract.require(lambda check_input: isinstance(check_input, bool), "check_input must be boolean")
@icontract.require(
    lambda precompute_is_array: isinstance(precompute_is_array, bool),
    "precompute_is_array must be boolean",
)
@icontract.ensure(
    lambda result, check_input, precompute_is_array: isinstance(result, bool)
    and result == (check_input and precompute_is_array),
    "Gram validation predicate must match check_input and precompute_is_array",
)
def cd_enet_path_gram_validation_required(
    check_input: bool, precompute_is_array: bool
) -> bool:
    """Return whether enet_path should validate a Gram matrix."""
    return check_input and precompute_is_array


@register_atom(witness_cd_enet_path_use_sparse_solver)
@icontract.require(lambda multi_output: isinstance(multi_output, bool), "multi_output must be boolean")
@icontract.require(lambda x_is_sparse: isinstance(x_is_sparse, bool), "x_is_sparse must be boolean")
@icontract.ensure(
    lambda result, multi_output, x_is_sparse: isinstance(result, bool)
    and result == ((not multi_output) and x_is_sparse),
    "sparse solver predicate must match the mono-output sparse branch",
)
def cd_enet_path_use_sparse_solver(multi_output: bool, x_is_sparse: bool) -> bool:
    """Return whether enet_path should dispatch to the sparse solver."""
    return (not multi_output) and x_is_sparse


@register_atom(witness_cd_enet_path_use_multi_task_solver)
@icontract.require(
    lambda use_sparse_solver: isinstance(use_sparse_solver, bool),
    "use_sparse_solver must be boolean",
)
@icontract.require(lambda multi_output: isinstance(multi_output, bool), "multi_output must be boolean")
@icontract.ensure(
    lambda result, use_sparse_solver, multi_output: isinstance(result, bool)
    and result == ((not use_sparse_solver) and multi_output),
    "multitask solver predicate must match the second dispatch branch",
)
def cd_enet_path_use_multi_task_solver(
    use_sparse_solver: bool, multi_output: bool
) -> bool:
    """Return whether enet_path should dispatch to the multitask solver."""
    return (not use_sparse_solver) and multi_output


@register_atom(witness_cd_enet_path_use_gram_solver)
@icontract.require(
    lambda use_sparse_solver: isinstance(use_sparse_solver, bool),
    "use_sparse_solver must be boolean",
)
@icontract.require(lambda multi_output: isinstance(multi_output, bool), "multi_output must be boolean")
@icontract.require(
    lambda precompute_is_array: isinstance(precompute_is_array, bool),
    "precompute_is_array must be boolean",
)
@icontract.ensure(
    lambda result, use_sparse_solver, multi_output, precompute_is_array: isinstance(result, bool)
    and result == ((not use_sparse_solver) and (not multi_output) and precompute_is_array),
    "Gram solver predicate must match the ndarray precompute branch",
)
def cd_enet_path_use_gram_solver(
    use_sparse_solver: bool, multi_output: bool, precompute_is_array: bool
) -> bool:
    """Return whether enet_path should dispatch to the Gram solver."""
    return (not use_sparse_solver) and (not multi_output) and precompute_is_array


@register_atom(witness_cd_enet_path_use_dense_solver)
@icontract.require(
    lambda use_sparse_solver: isinstance(use_sparse_solver, bool),
    "use_sparse_solver must be boolean",
)
@icontract.require(
    lambda use_multi_task_solver: isinstance(use_multi_task_solver, bool),
    "use_multi_task_solver must be boolean",
)
@icontract.require(
    lambda precompute_is_false: isinstance(precompute_is_false, bool),
    "precompute_is_false must be boolean",
)
@icontract.ensure(
    lambda result, use_sparse_solver, use_multi_task_solver, precompute_is_false: isinstance(result, bool)
    and result == ((not use_sparse_solver) and (not use_multi_task_solver) and precompute_is_false),
    "dense solver predicate must match the precompute is False branch",
)
def cd_enet_path_use_dense_solver(
    use_sparse_solver: bool, use_multi_task_solver: bool, precompute_is_false: bool
) -> bool:
    """Return whether enet_path should dispatch to the dense solver."""
    return (not use_sparse_solver) and (not use_multi_task_solver) and precompute_is_false


@register_atom(witness_cd_enet_path_invalid_precompute_message)
@icontract.ensure(
    lambda result, precompute: isinstance(result, str)
    and result
    == ("Precompute should be one of True, False, 'auto' or array-like. Got %r" % precompute),
    "invalid precompute message must match sklearn formatting",
)
def cd_enet_path_invalid_precompute_message(precompute: object) -> str:
    """Return the invalid precompute ValueError message used by enet_path."""
    return "Precompute should be one of True, False, 'auto' or array-like. Got %r" % precompute
