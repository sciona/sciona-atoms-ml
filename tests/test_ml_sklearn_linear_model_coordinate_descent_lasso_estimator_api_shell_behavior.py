from __future__ import annotations

import pytest
from icontract import ViolationError


def test_coordinate_descent_lasso_estimator_api_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_lasso_estimator_api_shell import (
        cd_lasso_constraints_without_l1_ratio,
        cd_lasso_fixed_l1_ratio,
        cd_lasso_path_name,
        cd_lasso_super_init_kwargs,
    )

    assert callable(cd_lasso_constraints_without_l1_ratio)
    assert callable(cd_lasso_path_name)
    assert callable(cd_lasso_fixed_l1_ratio)
    assert callable(cd_lasso_super_init_kwargs)


def test_coordinate_descent_lasso_estimator_api_shell_matches_sklearn_specialization() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_lasso_estimator_api_shell import (
        cd_lasso_constraints_without_l1_ratio,
        cd_lasso_fixed_l1_ratio,
        cd_lasso_path_name,
        cd_lasso_super_init_kwargs,
    )

    alpha_constraint = object()
    ratio_constraint = object()
    parent_constraints = {
        "alpha": alpha_constraint,
        "l1_ratio": ratio_constraint,
        "selection": "selection-constraint",
    }
    constraints = cd_lasso_constraints_without_l1_ratio(parent_constraints)
    assert constraints == {"alpha": alpha_constraint, "selection": "selection-constraint"}
    assert constraints["alpha"] is alpha_constraint
    assert "l1_ratio" not in constraints
    assert "l1_ratio" in parent_constraints

    assert cd_lasso_path_name("lasso") == "enet_path"
    assert cd_lasso_fixed_l1_ratio(0.25) == 1.0

    alpha = object()
    precompute = object()
    tol = object()
    random_state = object()
    selection = "random"
    kwargs = cd_lasso_super_init_kwargs(
        alpha,
        True,
        precompute,
        False,
        2000,
        tol,
        True,
        False,
        random_state,
        selection,
    )
    assert kwargs == {
        "alpha": alpha,
        "l1_ratio": 1.0,
        "fit_intercept": True,
        "precompute": precompute,
        "copy_X": False,
        "max_iter": 2000,
        "tol": tol,
        "warm_start": True,
        "positive": False,
        "random_state": random_state,
        "selection": selection,
    }
    assert kwargs["alpha"] is alpha
    assert kwargs["precompute"] is precompute
    assert kwargs["tol"] is tol
    assert kwargs["random_state"] is random_state


def test_coordinate_descent_lasso_estimator_api_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_lasso_estimator_api_shell import (
        cd_lasso_constraints_without_l1_ratio,
        cd_lasso_path_name,
        cd_lasso_super_init_kwargs,
    )

    with pytest.raises(ViolationError):
        cd_lasso_constraints_without_l1_ratio({"alpha": object()})

    with pytest.raises(ViolationError):
        cd_lasso_path_name("elastic_net")

    with pytest.raises(ViolationError):
        cd_lasso_super_init_kwargs(
            1.0,
            True,
            False,
            True,
            0,
            1e-4,
            False,
            False,
            None,
            "cyclic",
        )
