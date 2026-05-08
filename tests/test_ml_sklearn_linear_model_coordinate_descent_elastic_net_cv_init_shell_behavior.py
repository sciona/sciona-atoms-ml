from __future__ import annotations

import pytest
from icontract import ViolationError


def test_coordinate_descent_elastic_net_cv_init_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_elastic_net_cv_init_shell import (
        cd_elastic_net_cv_constraints,
        cd_elastic_net_cv_init_attributes,
    )

    assert callable(cd_elastic_net_cv_constraints)
    assert callable(cd_elastic_net_cv_init_attributes)


def test_coordinate_descent_elastic_net_cv_init_shell_matches_sklearn_api_setup() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_elastic_net_cv_init_shell import (
        cd_elastic_net_cv_constraints,
        cd_elastic_net_cv_init_attributes,
    )

    alpha_constraint = object()
    precompute_constraint = object()
    old_l1_ratio_constraint = object()
    l1_ratio_constraint = ["interval", "array-like"]
    parent_constraints = {
        "alpha": alpha_constraint,
        "precompute": precompute_constraint,
        "l1_ratio": old_l1_ratio_constraint,
    }

    constraints = cd_elastic_net_cv_constraints(parent_constraints, l1_ratio_constraint)
    assert set(constraints) == {"alpha", "precompute", "l1_ratio"}
    assert constraints["alpha"] is alpha_constraint
    assert constraints["precompute"] is precompute_constraint
    assert constraints["l1_ratio"] is l1_ratio_constraint

    alphas = object()
    precompute = object()
    cv = object()
    n_jobs = object()
    random_state = object()
    state = cd_elastic_net_cv_init_attributes(
        0.5,
        1e-3,
        100,
        alphas,
        True,
        precompute,
        1000,
        1e-4,
        cv,
        False,
        2,
        n_jobs,
        True,
        random_state,
        "random",
    )
    assert state == {
        "l1_ratio": 0.5,
        "eps": 1e-3,
        "n_alphas": 100,
        "alphas": alphas,
        "fit_intercept": True,
        "precompute": precompute,
        "max_iter": 1000,
        "tol": 1e-4,
        "cv": cv,
        "copy_X": False,
        "verbose": 2,
        "n_jobs": n_jobs,
        "positive": True,
        "random_state": random_state,
        "selection": "random",
    }
    assert state["alphas"] is alphas
    assert state["precompute"] is precompute
    assert state["cv"] is cv
    assert state["n_jobs"] is n_jobs
    assert state["random_state"] is random_state


def test_coordinate_descent_elastic_net_cv_init_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_elastic_net_cv_init_shell import (
        cd_elastic_net_cv_constraints,
        cd_elastic_net_cv_init_attributes,
    )

    with pytest.raises(ViolationError):
        cd_elastic_net_cv_constraints([], object())  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_elastic_net_cv_constraints({"alpha": object()}, None)

    with pytest.raises(ViolationError):
        cd_elastic_net_cv_init_attributes(
            0.5, 1e-3, 0, None, True, "auto", 1000, 1e-4, None, True, 0, None, False, None, "cyclic"
        )

    with pytest.raises(ViolationError):
        cd_elastic_net_cv_init_attributes(
            0.5, 1e-3, 100, None, "yes", "auto", 1000, 1e-4, None, True, 0, None, False, None, "cyclic"
        )  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_elastic_net_cv_init_attributes(
            0.5, 1e-3, 100, None, True, "auto", 0, 1e-4, None, True, 0, None, False, None, "cyclic"
        )

    with pytest.raises(ViolationError):
        cd_elastic_net_cv_init_attributes(
            0.5, 1e-3, 100, None, True, "auto", 1000, 1e-4, None, True, 0, None, "no", None, "cyclic"
        )  # type: ignore[arg-type]
