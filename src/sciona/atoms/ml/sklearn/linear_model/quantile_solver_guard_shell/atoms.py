"""Sklearn QuantileRegressor solver guard atoms."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_quantile_interior_point_removed_guard,
    witness_quantile_interior_point_removed_message,
    witness_quantile_solver_options_payload,
    witness_quantile_sparse_solver_guard,
    witness_quantile_sparse_solver_message,
)

_SPARSE_COMPATIBLE_SOLVERS = {"highs", "highs-ds", "highs-ipm"}


def _solver_valid(solver: str) -> bool:
    return bool(isinstance(solver, str) and solver)


def _bool_valid(value: bool) -> bool:
    return isinstance(value, bool)


def _options_valid(options: Mapping[str, Any] | None) -> bool:
    return options is None or isinstance(options, Mapping)


@register_atom(witness_quantile_interior_point_removed_guard)
@icontract.require(lambda solver: _solver_valid(solver), "solver must be a nonempty string")
@icontract.require(lambda scipy_at_least_1_11: _bool_valid(scipy_at_least_1_11), "scipy version predicate must be boolean")
@icontract.ensure(
    lambda result, solver, scipy_at_least_1_11: result is (solver == "interior-point" and scipy_at_least_1_11),
    "interior-point removal guard must match solver and SciPy version predicate",
)
def quantile_interior_point_removed_guard(solver: str, scipy_at_least_1_11: bool) -> bool:
    """Return whether QuantileRegressor must reject removed interior-point."""
    return solver == "interior-point" and scipy_at_least_1_11


@register_atom(witness_quantile_interior_point_removed_message)
@icontract.require(lambda solver: solver == "interior-point", "solver must be interior-point")
@icontract.ensure(
    lambda result, solver: result == f"Solver {solver} is not anymore available in SciPy >= 1.11.0.",
    "interior-point removal message must match sklearn wording",
)
def quantile_interior_point_removed_message(solver: str) -> str:
    """Return the QuantileRegressor interior-point removal error message."""
    return f"Solver {solver} is not anymore available in SciPy >= 1.11.0."


@register_atom(witness_quantile_sparse_solver_guard)
@icontract.require(lambda is_sparse: _bool_valid(is_sparse), "is_sparse must be boolean")
@icontract.require(lambda solver: _solver_valid(solver), "solver must be a nonempty string")
@icontract.ensure(
    lambda result, is_sparse, solver: result is (is_sparse and solver not in _SPARSE_COMPATIBLE_SOLVERS),
    "sparse solver guard must reject sparse X unless solver is a highs variant",
)
def quantile_sparse_solver_guard(is_sparse: bool, solver: str) -> bool:
    """Return whether sparse X must reject the selected QuantileRegressor solver."""
    return is_sparse and solver not in _SPARSE_COMPATIBLE_SOLVERS


@register_atom(witness_quantile_sparse_solver_message)
@icontract.require(lambda solver: _solver_valid(solver), "solver must be a nonempty string")
@icontract.require(
    lambda solver: solver not in _SPARSE_COMPATIBLE_SOLVERS,
    "solver must be unsupported for sparse X",
)
@icontract.ensure(
    lambda result, solver: result == f"Solver {solver} does not support sparse X. Use solver 'highs' for example.",
    "sparse solver message must match sklearn wording",
)
def quantile_sparse_solver_message(solver: str) -> str:
    """Return the QuantileRegressor sparse-X unsupported-solver error message."""
    return f"Solver {solver} does not support sparse X. Use solver 'highs' for example."


@register_atom(witness_quantile_solver_options_payload)
@icontract.require(lambda solver_options: _options_valid(solver_options), "solver_options must be a mapping or None")
@icontract.require(lambda solver: _solver_valid(solver), "solver must be a nonempty string")
@icontract.ensure(
    lambda result, solver_options, solver: (
        result == {"lstsq": True} if solver_options is None and solver == "interior-point" else result is solver_options
    ),
    "solver_options payload must default interior-point or preserve the supplied object",
)
def quantile_solver_options_payload(
    solver_options: Mapping[str, Any] | None,
    solver: str,
) -> Mapping[str, Any] | None:
    """Return the solver_options payload selected before QuantileRegressor linprog."""
    if solver_options is None and solver == "interior-point":
        return {"lstsq": True}
    return solver_options
