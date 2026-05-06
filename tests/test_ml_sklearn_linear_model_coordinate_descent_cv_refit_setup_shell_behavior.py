from __future__ import annotations

import pytest
from icontract import ViolationError


def test_coordinate_descent_cv_refit_setup_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_refit_setup_shell import (
        cd_cv_refit_common_params,
        cd_cv_refit_copy_x,
        cd_cv_refit_fit_call_uses_sample_weight,
        cd_cv_refit_model_alpha,
        cd_cv_refit_model_l1_ratio,
        cd_cv_refit_precompute_auto_guard_required,
        cd_cv_refit_precompute_value,
    )

    assert callable(cd_cv_refit_common_params)
    assert callable(cd_cv_refit_model_alpha)
    assert callable(cd_cv_refit_model_l1_ratio)
    assert callable(cd_cv_refit_copy_x)
    assert callable(cd_cv_refit_precompute_auto_guard_required)
    assert callable(cd_cv_refit_precompute_value)
    assert callable(cd_cv_refit_fit_call_uses_sample_weight)


def test_coordinate_descent_cv_refit_setup_shell_matches_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_refit_setup_shell import (
        cd_cv_refit_common_params,
        cd_cv_refit_copy_x,
        cd_cv_refit_fit_call_uses_sample_weight,
        cd_cv_refit_model_alpha,
        cd_cv_refit_model_l1_ratio,
        cd_cv_refit_precompute_auto_guard_required,
        cd_cv_refit_precompute_value,
    )

    common = cd_cv_refit_common_params(
        {"alpha": 1.0, "cv": 5, "tol": 1e-4},
        {"alpha", "tol"},
    )
    assert common == {"alpha": 1.0, "tol": 1e-4}
    assert cd_cv_refit_model_alpha(0.25) == pytest.approx(0.25)
    token = object()
    assert cd_cv_refit_model_l1_ratio(token) is token
    assert cd_cv_refit_copy_x(False) is False
    assert cd_cv_refit_precompute_auto_guard_required("auto") is True
    assert cd_cv_refit_precompute_value("auto", True) is False
    assert cd_cv_refit_precompute_value(True, False) is True
    assert cd_cv_refit_fit_call_uses_sample_weight([1.0, 2.0]) is True


def test_coordinate_descent_cv_refit_setup_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_refit_setup_shell import (
        cd_cv_refit_common_params,
        cd_cv_refit_model_alpha,
    )

    with pytest.raises(ViolationError):
        cd_cv_refit_model_alpha(float("nan"))

    with pytest.raises(ViolationError):
        cd_cv_refit_common_params([], {"alpha"})  # type: ignore[arg-type]
