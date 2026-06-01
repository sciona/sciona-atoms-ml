"""Sklearn QuantileRegressor linprog callback-boundary atoms."""

from __future__ import annotations

from collections.abc import Mapping

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_quantile_linprog_callback_payload,
    witness_quantile_linprog_solution,
)

_LINPROG_PAYLOAD_KEYS = {"c", "A_eq", "b_eq", "method", "options"}


def _nonempty_string(value: str) -> bool:
    return bool(isinstance(value, str) and value)


def _not_none(value: object) -> bool:
    return value is not None


def _options_valid(value: object) -> bool:
    return value is None or isinstance(value, Mapping)


def _linprog_payload_valid(
    result: Mapping[str, object],
    c: object,
    A_eq: object,
    b_eq: object,
    solver: str,
    solver_options: object,
) -> bool:
    return bool(
        set(result) == _LINPROG_PAYLOAD_KEYS
        and result["c"] is c
        and result["A_eq"] is A_eq
        and result["b_eq"] is b_eq
        and result["method"] == solver
        and result["options"] is solver_options
    )


def _has_solution_x(result: object) -> bool:
    return hasattr(result, "x") and getattr(result, "x") is not None


@register_atom(witness_quantile_linprog_callback_payload)
@icontract.require(lambda c: _not_none(c), "c must be supplied")
@icontract.require(lambda A_eq: _not_none(A_eq), "A_eq must be supplied")
@icontract.require(lambda b_eq: _not_none(b_eq), "b_eq must be supplied")
@icontract.require(lambda solver: _nonempty_string(solver), "solver must be a nonempty string")
@icontract.require(lambda solver_options: _options_valid(solver_options), "solver_options must be a mapping or None")
@icontract.ensure(
    lambda result, c, A_eq, b_eq, solver, solver_options: _linprog_payload_valid(
        result,
        c,
        A_eq,
        b_eq,
        solver,
        solver_options,
    ),
    "linprog callback payload must preserve sklearn keyword identities",
)
def quantile_linprog_callback_payload(
    c: object,
    A_eq: object,
    b_eq: object,
    solver: str,
    solver_options: Mapping[str, object] | None,
) -> dict[str, object]:
    """Return the exact keyword payload passed to scipy.optimize.linprog."""
    return {"c": c, "A_eq": A_eq, "b_eq": b_eq, "method": solver, "options": solver_options}


@register_atom(witness_quantile_linprog_solution)
@icontract.require(lambda result_obj: _has_solution_x(result_obj), "result must expose a non-None x attribute")
@icontract.ensure(lambda result, result_obj: result is result_obj.x, "solution extraction must preserve result.x identity")
def quantile_linprog_solution(result_obj: object) -> object:
    """Return the raw linprog solution stored on result.x."""
    return result_obj.x
