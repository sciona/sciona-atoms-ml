from __future__ import annotations

import pytest
from icontract import ViolationError


def test_coordinate_descent_multitask_cv_api_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_multitask_cv_api_shell import (
        cd_multitask_elastic_net_cv_constraints,
        cd_multitask_elastic_net_cv_init_attributes,
        cd_multitask_lasso_cv_constraints_without_unsupported,
        cd_multitask_lasso_cv_super_init_kwargs,
    )

    assert callable(cd_multitask_elastic_net_cv_constraints)
    assert callable(cd_multitask_elastic_net_cv_init_attributes)
    assert callable(cd_multitask_lasso_cv_constraints_without_unsupported)
    assert callable(cd_multitask_lasso_cv_super_init_kwargs)


def test_coordinate_descent_multitask_cv_api_shell_matches_sklearn_api_setup() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_multitask_cv_api_shell import (
        cd_multitask_elastic_net_cv_constraints,
        cd_multitask_elastic_net_cv_init_attributes,
        cd_multitask_lasso_cv_constraints_without_unsupported,
        cd_multitask_lasso_cv_super_init_kwargs,
    )

    alpha_constraint = object()
    precompute_constraint = object()
    positive_constraint = object()
    old_l1_ratio_constraint = object()
    l1_ratio_constraint = ["interval", "array-like"]
    parent_constraints = {
        "alpha": alpha_constraint,
        "precompute": precompute_constraint,
        "positive": positive_constraint,
        "l1_ratio": old_l1_ratio_constraint,
    }

    enet_constraints = cd_multitask_elastic_net_cv_constraints(
        parent_constraints, l1_ratio_constraint
    )
    assert set(enet_constraints) == {"alpha", "l1_ratio"}
    assert enet_constraints["alpha"] is alpha_constraint
    assert enet_constraints["l1_ratio"] is l1_ratio_constraint
    assert "precompute" not in enet_constraints
    assert "positive" not in enet_constraints

    alphas = object()
    cv = object()
    n_jobs = object()
    random_state = object()
    enet_state = cd_multitask_elastic_net_cv_init_attributes(
        0.5,
        1e-3,
        100,
        alphas,
        True,
        1000,
        1e-4,
        cv,
        False,
        2,
        n_jobs,
        random_state,
        "random",
    )
    assert enet_state == {
        "l1_ratio": 0.5,
        "eps": 1e-3,
        "n_alphas": 100,
        "alphas": alphas,
        "fit_intercept": True,
        "max_iter": 1000,
        "tol": 1e-4,
        "cv": cv,
        "copy_X": False,
        "verbose": 2,
        "n_jobs": n_jobs,
        "random_state": random_state,
        "selection": "random",
    }
    assert enet_state["alphas"] is alphas
    assert enet_state["cv"] is cv
    assert enet_state["n_jobs"] is n_jobs
    assert enet_state["random_state"] is random_state

    lasso_constraints = cd_multitask_lasso_cv_constraints_without_unsupported(
        parent_constraints
    )
    assert set(lasso_constraints) == {"alpha", "l1_ratio"}
    assert lasso_constraints["alpha"] is alpha_constraint
    assert lasso_constraints["l1_ratio"] is old_l1_ratio_constraint
    assert "precompute" not in lasso_constraints
    assert "positive" not in lasso_constraints

    lasso_kwargs = cd_multitask_lasso_cv_super_init_kwargs(
        1e-2,
        25,
        alphas,
        False,
        500,
        1e-5,
        True,
        cv,
        False,
        n_jobs,
        random_state,
        "cyclic",
    )
    assert lasso_kwargs == {
        "eps": 1e-2,
        "n_alphas": 25,
        "alphas": alphas,
        "fit_intercept": False,
        "max_iter": 500,
        "tol": 1e-5,
        "copy_X": True,
        "cv": cv,
        "verbose": False,
        "n_jobs": n_jobs,
        "random_state": random_state,
        "selection": "cyclic",
    }
    assert lasso_kwargs["alphas"] is alphas
    assert lasso_kwargs["cv"] is cv
    assert lasso_kwargs["n_jobs"] is n_jobs
    assert lasso_kwargs["random_state"] is random_state


def test_coordinate_descent_multitask_cv_api_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_multitask_cv_api_shell import (
        cd_multitask_elastic_net_cv_constraints,
        cd_multitask_elastic_net_cv_init_attributes,
        cd_multitask_lasso_cv_constraints_without_unsupported,
        cd_multitask_lasso_cv_super_init_kwargs,
    )

    with pytest.raises(ViolationError):
        cd_multitask_elastic_net_cv_constraints({"precompute": object()}, object())

    with pytest.raises(ViolationError):
        cd_multitask_elastic_net_cv_init_attributes(
            0.5, 1e-3, 0, None, True, 1000, 1e-4, None, True, 0, None, None, "cyclic"
        )

    with pytest.raises(ViolationError):
        cd_multitask_elastic_net_cv_init_attributes(
            0.5, 1e-3, 100, None, "yes", 1000, 1e-4, None, True, 0, None, None, "cyclic"
        )  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_multitask_lasso_cv_constraints_without_unsupported({"positive": object()})

    with pytest.raises(ViolationError):
        cd_multitask_lasso_cv_super_init_kwargs(
            1e-3, 100, None, True, 0, 1e-4, True, None, False, None, None, "cyclic"
        )

    with pytest.raises(ViolationError):
        cd_multitask_lasso_cv_super_init_kwargs(
            1e-3, 100, None, True, 1000, 1e-4, True, None, False, None, None, object()
        )  # type: ignore[arg-type]
