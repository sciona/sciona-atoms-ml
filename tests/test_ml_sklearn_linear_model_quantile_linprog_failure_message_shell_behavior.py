from __future__ import annotations

import pytest
from icontract import ViolationError


def test_quantile_linprog_failure_message_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.quantile_linprog_failure_message_shell import (
        quantile_linprog_failure_guard,
        quantile_linprog_failure_message,
        quantile_linprog_failure_reason,
    )

    assert callable(quantile_linprog_failure_guard)
    assert callable(quantile_linprog_failure_reason)
    assert callable(quantile_linprog_failure_message)


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (1, "Iteration limit reached."),
        (2, "Problem appears to be infeasible."),
        (3, "Problem appears to be unbounded."),
        (4, "Numerical difficulties encountered."),
        (9, "unknown reason"),
    ],
)
def test_quantile_linprog_failure_reason_matches_source_mapping(status: int, reason: str) -> None:
    from sciona.atoms.ml.sklearn.linear_model.quantile_linprog_failure_message_shell import (
        quantile_linprog_failure_reason,
    )

    assert quantile_linprog_failure_reason(status) == reason


def test_quantile_linprog_failure_message_matches_source_payload() -> None:
    from sciona.atoms.ml.sklearn.linear_model.quantile_linprog_failure_message_shell import (
        quantile_linprog_failure_message,
    )

    result_message = "HiGHS Status 8: model_status is Infeasible"
    expected = (
        "Linear programming for QuantileRegressor did not succeed.\n"
        "Status is 2: "
        "Problem appears to be infeasible."
        "\n"
        "Result message of linprog:\n"
        + result_message
    )

    assert quantile_linprog_failure_message(2, result_message) == expected


def test_quantile_linprog_failure_guard_matches_warning_branch_condition() -> None:
    from sciona.atoms.ml.sklearn.linear_model.quantile_linprog_failure_message_shell import (
        quantile_linprog_failure_guard,
    )

    assert quantile_linprog_failure_guard(False) is True
    assert quantile_linprog_failure_guard(True) is False


def test_quantile_linprog_failure_message_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.quantile_linprog_failure_message_shell import (
        quantile_linprog_failure_guard,
        quantile_linprog_failure_message,
        quantile_linprog_failure_reason,
    )

    with pytest.raises(ViolationError):
        quantile_linprog_failure_guard(0)

    with pytest.raises(ViolationError):
        quantile_linprog_failure_reason(True)

    with pytest.raises(ViolationError):
        quantile_linprog_failure_message(1.5, "message")

    with pytest.raises(ViolationError):
        quantile_linprog_failure_message(1, None)
