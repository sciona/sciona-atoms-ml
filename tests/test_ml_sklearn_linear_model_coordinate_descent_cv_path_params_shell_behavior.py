from __future__ import annotations

import pytest
from icontract import ViolationError


def test_coordinate_descent_cv_path_params_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_path_params_shell import (
        cd_cv_parallel_copy_x_override_required,
        cd_cv_path_params_copy_x,
        cd_cv_path_params_cv_removed,
        cd_cv_path_params_fit_intercept_removed,
        cd_cv_path_params_n_alphas,
        cd_cv_path_params_n_jobs_removed,
        cd_cv_resolved_path_copy_x,
    )

    assert callable(cd_cv_path_params_fit_intercept_removed)
    assert callable(cd_cv_path_params_cv_removed)
    assert callable(cd_cv_path_params_n_jobs_removed)
    assert callable(cd_cv_path_params_n_alphas)
    assert callable(cd_cv_path_params_copy_x)
    assert callable(cd_cv_parallel_copy_x_override_required)
    assert callable(cd_cv_resolved_path_copy_x)


def test_coordinate_descent_cv_path_params_shell_matches_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_path_params_shell import (
        cd_cv_parallel_copy_x_override_required,
        cd_cv_path_params_copy_x,
        cd_cv_path_params_cv_removed,
        cd_cv_path_params_fit_intercept_removed,
        cd_cv_path_params_n_alphas,
        cd_cv_path_params_n_jobs_removed,
        cd_cv_resolved_path_copy_x,
    )

    path_params = {"tol": 1e-4}
    assert cd_cv_path_params_fit_intercept_removed(path_params) is True
    assert cd_cv_path_params_cv_removed(path_params) is True
    assert cd_cv_path_params_n_jobs_removed(path_params) is True
    assert cd_cv_path_params_n_alphas(17) == {"n_alphas": 17}
    assert cd_cv_path_params_copy_x(True) is True
    assert cd_cv_parallel_copy_x_override_required(True) is True
    assert cd_cv_resolved_path_copy_x(True, True) is False
    assert cd_cv_resolved_path_copy_x(False, False) is False


def test_coordinate_descent_cv_path_params_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_path_params_shell import (
        cd_cv_path_params_n_alphas,
        cd_cv_resolved_path_copy_x,
    )

    with pytest.raises(ViolationError):
        cd_cv_path_params_n_alphas(0)

    with pytest.raises(ViolationError):
        cd_cv_resolved_path_copy_x(True, None)  # type: ignore[arg-type]
