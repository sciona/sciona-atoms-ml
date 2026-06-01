from __future__ import annotations

from types import SimpleNamespace

import pytest
from icontract import ViolationError


def test_quantile_linprog_callback_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.quantile_linprog_callback_shell import (
        quantile_linprog_callback_payload,
        quantile_linprog_solution,
    )

    assert callable(quantile_linprog_callback_payload)
    assert callable(quantile_linprog_solution)


def test_quantile_linprog_callback_payload_preserves_solver_arguments() -> None:
    from sciona.atoms.ml.sklearn.linear_model.quantile_linprog_callback_shell import quantile_linprog_callback_payload

    c = object()
    A_eq = object()
    b_eq = object()
    options = {"presolve": False}

    payload = quantile_linprog_callback_payload(c, A_eq, b_eq, "highs", options)

    assert payload == {"c": c, "A_eq": A_eq, "b_eq": b_eq, "method": "highs", "options": options}
    assert payload["c"] is c
    assert payload["A_eq"] is A_eq
    assert payload["b_eq"] is b_eq
    assert payload["options"] is options


def test_quantile_linprog_callback_payload_preserves_none_options() -> None:
    from sciona.atoms.ml.sklearn.linear_model.quantile_linprog_callback_shell import quantile_linprog_callback_payload

    payload = quantile_linprog_callback_payload(object(), object(), object(), "highs-ipm", None)

    assert payload["method"] == "highs-ipm"
    assert payload["options"] is None


def test_quantile_linprog_solution_extracts_raw_result_x() -> None:
    from sciona.atoms.ml.sklearn.linear_model.quantile_linprog_callback_shell import quantile_linprog_solution

    solution = object()
    result = SimpleNamespace(x=solution)

    assert quantile_linprog_solution(result) is solution


def test_quantile_linprog_callback_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.quantile_linprog_callback_shell import (
        quantile_linprog_callback_payload,
        quantile_linprog_solution,
    )

    with pytest.raises(ViolationError):
        quantile_linprog_callback_payload(None, object(), object(), "highs", None)

    with pytest.raises(ViolationError):
        quantile_linprog_callback_payload(object(), object(), object(), "", None)

    with pytest.raises(ViolationError):
        quantile_linprog_callback_payload(object(), object(), object(), "highs", ["bad"])

    with pytest.raises(ViolationError):
        quantile_linprog_solution(SimpleNamespace(x=None))
