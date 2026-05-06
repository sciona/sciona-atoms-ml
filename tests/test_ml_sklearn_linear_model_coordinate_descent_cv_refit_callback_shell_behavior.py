from __future__ import annotations

import pytest
from icontract import ViolationError


def test_coordinate_descent_cv_refit_callback_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_refit_callback_shell import (
        cd_cv_refit_fitted_model,
        cd_cv_refit_set_params_kwargs,
        cd_cv_refit_set_params_result,
        cd_cv_refit_weighted_fit_kwargs,
    )

    assert callable(cd_cv_refit_set_params_kwargs)
    assert callable(cd_cv_refit_set_params_result)
    assert callable(cd_cv_refit_weighted_fit_kwargs)
    assert callable(cd_cv_refit_fitted_model)


def test_coordinate_descent_cv_refit_callback_shell_matches_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_refit_callback_shell import (
        cd_cv_refit_fitted_model,
        cd_cv_refit_set_params_kwargs,
        cd_cv_refit_set_params_result,
        cd_cv_refit_weighted_fit_kwargs,
    )

    common_params = {"tol": 1e-4, "max_iter": 1000}
    assert cd_cv_refit_set_params_kwargs(common_params) == common_params

    model = object()
    assert cd_cv_refit_set_params_result(model) is model
    assert cd_cv_refit_weighted_fit_kwargs([1.0, 2.0]) == {"sample_weight": [1.0, 2.0]}
    assert cd_cv_refit_fitted_model(model) is model


def test_coordinate_descent_cv_refit_callback_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_refit_callback_shell import (
        cd_cv_refit_set_params_kwargs,
        cd_cv_refit_weighted_fit_kwargs,
    )

    with pytest.raises(ViolationError):
        cd_cv_refit_set_params_kwargs([("tol", 1e-4)])

    with pytest.raises(ViolationError):
        cd_cv_refit_weighted_fit_kwargs(None)
