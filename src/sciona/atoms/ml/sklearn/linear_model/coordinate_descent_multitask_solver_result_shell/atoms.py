"""Sklearn coordinate-descent multitask solver result atoms."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_multitask_set_intercept_args,
    witness_cd_multitask_solver_result_coef,
    witness_cd_multitask_solver_result_dual_gap,
    witness_cd_multitask_solver_result_eps,
    witness_cd_multitask_solver_result_n_iter,
)


def _solver_result(value: object) -> bool:
    return isinstance(value, tuple) and len(value) == 4


@register_atom(witness_cd_multitask_solver_result_coef)
@icontract.require(lambda solver_result: _solver_result(solver_result), "solver_result must be a four-item tuple")
@icontract.ensure(
    lambda result, solver_result: result is solver_result[0],
    "coefficient result must be solver_result[0]",
)
def cd_multitask_solver_result_coef(solver_result: tuple[object, object, object, object]) -> object:
    """Return the fitted coefficient payload from the multitask solver result."""
    return solver_result[0]


@register_atom(witness_cd_multitask_solver_result_dual_gap)
@icontract.require(lambda solver_result: _solver_result(solver_result), "solver_result must be a four-item tuple")
@icontract.ensure(
    lambda result, solver_result: result is solver_result[1],
    "dual-gap result must be solver_result[1]",
)
def cd_multitask_solver_result_dual_gap(solver_result: tuple[object, object, object, object]) -> object:
    """Return the raw dual-gap payload from the multitask solver result."""
    return solver_result[1]


@register_atom(witness_cd_multitask_solver_result_eps)
@icontract.require(lambda solver_result: _solver_result(solver_result), "solver_result must be a four-item tuple")
@icontract.ensure(
    lambda result, solver_result: result is solver_result[2],
    "eps result must be solver_result[2]",
)
def cd_multitask_solver_result_eps(solver_result: tuple[object, object, object, object]) -> object:
    """Return the eps payload from the multitask solver result."""
    return solver_result[2]


@register_atom(witness_cd_multitask_solver_result_n_iter)
@icontract.require(lambda solver_result: _solver_result(solver_result), "solver_result must be a four-item tuple")
@icontract.ensure(
    lambda result, solver_result: result is solver_result[3],
    "iteration result must be solver_result[3]",
)
def cd_multitask_solver_result_n_iter(solver_result: tuple[object, object, object, object]) -> object:
    """Return the n_iter payload from the multitask solver result."""
    return solver_result[3]


@register_atom(witness_cd_multitask_set_intercept_args)
@icontract.require(lambda X_offset: X_offset is not None, "X_offset must be provided")
@icontract.require(lambda y_offset: y_offset is not None, "y_offset must be provided")
@icontract.require(lambda X_scale: X_scale is not None, "X_scale must be provided")
@icontract.ensure(
    lambda result, X_offset, y_offset, X_scale: isinstance(result, tuple)
    and len(result) == 3
    and result[0] is X_offset
    and result[1] is y_offset
    and result[2] is X_scale,
    "set-intercept args must match MultiTaskElasticNet.fit call order",
)
def cd_multitask_set_intercept_args(
    X_offset: object,
    y_offset: object,
    X_scale: object,
) -> tuple[object, object, object]:
    """Return the positional payload for MultiTaskElasticNet.fit `_set_intercept`."""
    return X_offset, y_offset, X_scale
