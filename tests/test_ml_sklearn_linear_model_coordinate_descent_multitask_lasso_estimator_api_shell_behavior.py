from __future__ import annotations

import pytest
from icontract import ViolationError


def test_coordinate_descent_multitask_lasso_estimator_api_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_multitask_lasso_estimator_api_shell import (
        cd_multitask_lasso_constraints_without_l1_ratio,
        cd_multitask_lasso_fixed_l1_ratio,
        cd_multitask_lasso_init_attributes,
    )

    assert callable(cd_multitask_lasso_constraints_without_l1_ratio)
    assert callable(cd_multitask_lasso_fixed_l1_ratio)
    assert callable(cd_multitask_lasso_init_attributes)


def test_coordinate_descent_multitask_lasso_estimator_api_shell_matches_sklearn_specialization() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_multitask_lasso_estimator_api_shell import (
        cd_multitask_lasso_constraints_without_l1_ratio,
        cd_multitask_lasso_fixed_l1_ratio,
        cd_multitask_lasso_init_attributes,
    )

    alpha_constraint = object()
    ratio_constraint = object()
    parent_constraints = {
        "alpha": alpha_constraint,
        "l1_ratio": ratio_constraint,
        "selection": "selection-constraint",
    }
    constraints = cd_multitask_lasso_constraints_without_l1_ratio(parent_constraints)
    assert constraints == {"alpha": alpha_constraint, "selection": "selection-constraint"}
    assert constraints["alpha"] is alpha_constraint
    assert "l1_ratio" not in constraints
    assert "l1_ratio" in parent_constraints

    assert cd_multitask_lasso_fixed_l1_ratio(0.25) == 1.0

    alpha = object()
    tol = object()
    random_state = object()
    selection = "random"
    attrs = cd_multitask_lasso_init_attributes(
        alpha,
        True,
        False,
        2000,
        tol,
        True,
        random_state,
        selection,
    )
    assert attrs == {
        "alpha": alpha,
        "fit_intercept": True,
        "max_iter": 2000,
        "copy_X": False,
        "tol": tol,
        "warm_start": True,
        "l1_ratio": 1.0,
        "random_state": random_state,
        "selection": selection,
    }
    assert attrs["alpha"] is alpha
    assert attrs["tol"] is tol
    assert attrs["random_state"] is random_state


def test_coordinate_descent_multitask_lasso_estimator_api_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_multitask_lasso_estimator_api_shell import (
        cd_multitask_lasso_constraints_without_l1_ratio,
        cd_multitask_lasso_init_attributes,
    )

    with pytest.raises(ViolationError):
        cd_multitask_lasso_constraints_without_l1_ratio({"alpha": object()})

    with pytest.raises(ViolationError):
        cd_multitask_lasso_init_attributes(
            1.0,
            True,
            True,
            0,
            1e-4,
            False,
            None,
            "cyclic",
        )
