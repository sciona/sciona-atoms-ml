from __future__ import annotations

import pytest
from icontract import ViolationError


def test_coordinate_descent_cv_base_init_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_base_init_shell import (
        cd_cv_base_init_attributes,
    )

    assert callable(cd_cv_base_init_attributes)


def test_coordinate_descent_cv_base_init_shell_matches_sklearn_assignments() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_base_init_shell import (
        cd_cv_base_init_attributes,
    )

    alphas = object()
    precompute = object()
    cv = object()
    n_jobs = object()
    random_state = object()
    state = cd_cv_base_init_attributes(
        1e-3,
        100,
        alphas,
        True,
        precompute,
        1000,
        1e-4,
        False,
        cv,
        2,
        n_jobs,
        True,
        random_state,
        "random",
    )
    assert state == {
        "eps": 1e-3,
        "n_alphas": 100,
        "alphas": alphas,
        "fit_intercept": True,
        "precompute": precompute,
        "max_iter": 1000,
        "tol": 1e-4,
        "copy_X": False,
        "cv": cv,
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


def test_coordinate_descent_cv_base_init_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_base_init_shell import (
        cd_cv_base_init_attributes,
    )

    with pytest.raises(ViolationError):
        cd_cv_base_init_attributes(
            1e-3, 0, None, True, "auto", 1000, 1e-4, True, None, False, None, False, None, "cyclic"
        )

    with pytest.raises(ViolationError):
        cd_cv_base_init_attributes(
            1e-3, 100, None, "yes", "auto", 1000, 1e-4, True, None, False, None, False, None, "cyclic"
        )  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_cv_base_init_attributes(
            1e-3, 100, None, True, "auto", 0, 1e-4, True, None, False, None, False, None, "cyclic"
        )

    with pytest.raises(ViolationError):
        cd_cv_base_init_attributes(
            1e-3, 100, None, True, "auto", 1000, 1e-4, "yes", None, False, None, False, None, "cyclic"
        )  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_cv_base_init_attributes(
            1e-3, 100, None, True, "auto", 1000, 1e-4, True, None, False, None, "no", None, "cyclic"
        )  # type: ignore[arg-type]
