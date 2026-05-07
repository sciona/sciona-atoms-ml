from __future__ import annotations

import pytest
from icontract import ViolationError


def test_coordinate_descent_multitask_elastic_net_api_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_multitask_elastic_net_api_shell import (
        cd_multitask_elastic_net_constraints_without_unsupported,
        cd_multitask_elastic_net_init_attributes,
    )

    assert callable(cd_multitask_elastic_net_constraints_without_unsupported)
    assert callable(cd_multitask_elastic_net_init_attributes)


def test_coordinate_descent_multitask_elastic_net_api_shell_matches_sklearn_api() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_multitask_elastic_net_api_shell import (
        cd_multitask_elastic_net_constraints_without_unsupported,
        cd_multitask_elastic_net_init_attributes,
    )

    alpha_constraint = object()
    ratio_constraint = object()
    parent_constraints = {
        "alpha": alpha_constraint,
        "l1_ratio": ratio_constraint,
        "precompute": object(),
        "positive": object(),
        "selection": "selection-constraint",
    }
    constraints = cd_multitask_elastic_net_constraints_without_unsupported(parent_constraints)
    assert constraints == {
        "alpha": alpha_constraint,
        "l1_ratio": ratio_constraint,
        "selection": "selection-constraint",
    }
    assert constraints["alpha"] is alpha_constraint
    assert constraints["l1_ratio"] is ratio_constraint
    assert "precompute" not in constraints
    assert "positive" not in constraints
    assert "precompute" in parent_constraints
    assert "positive" in parent_constraints

    alpha = object()
    l1_ratio = object()
    tol = object()
    random_state = object()
    attrs = cd_multitask_elastic_net_init_attributes(
        alpha,
        l1_ratio,
        True,
        False,
        2000,
        tol,
        True,
        random_state,
        "random",
    )
    assert attrs == {
        "l1_ratio": l1_ratio,
        "alpha": alpha,
        "fit_intercept": True,
        "max_iter": 2000,
        "copy_X": False,
        "tol": tol,
        "warm_start": True,
        "random_state": random_state,
        "selection": "random",
    }
    assert attrs["alpha"] is alpha
    assert attrs["l1_ratio"] is l1_ratio
    assert attrs["tol"] is tol
    assert attrs["random_state"] is random_state


def test_coordinate_descent_multitask_elastic_net_api_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_multitask_elastic_net_api_shell import (
        cd_multitask_elastic_net_constraints_without_unsupported,
        cd_multitask_elastic_net_init_attributes,
    )

    with pytest.raises(ViolationError):
        cd_multitask_elastic_net_constraints_without_unsupported(
            {"alpha": object(), "positive": object()}
        )

    with pytest.raises(ViolationError):
        cd_multitask_elastic_net_init_attributes(
            1.0,
            0.5,
            True,
            True,
            0,
            1e-4,
            False,
            None,
            "cyclic",
        )
