from __future__ import annotations

import pytest
from icontract import ViolationError


def test_coordinate_descent_cv_estimator_params_callback_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_estimator_params_callback_shell import (
        cd_cv_get_estimator_result,
        cd_cv_model_get_params_result,
        cd_cv_model_param_names,
        cd_cv_path_get_params_result,
        cd_cv_refit_get_params_result,
    )

    assert callable(cd_cv_get_estimator_result)
    assert callable(cd_cv_path_get_params_result)
    assert callable(cd_cv_refit_get_params_result)
    assert callable(cd_cv_model_get_params_result)
    assert callable(cd_cv_model_param_names)


def test_coordinate_descent_cv_estimator_params_callback_shell_matches_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_estimator_params_callback_shell import (
        cd_cv_get_estimator_result,
        cd_cv_model_get_params_result,
        cd_cv_model_param_names,
        cd_cv_path_get_params_result,
        cd_cv_refit_get_params_result,
    )

    model = object()
    assert cd_cv_get_estimator_result(model) is model

    alpha_payload = object()
    self_params = {
        "alpha": alpha_payload,
        "cv": 5,
        "n_jobs": None,
    }
    path_params = cd_cv_path_get_params_result(self_params)
    assert path_params == self_params
    assert path_params is not self_params
    assert path_params["alpha"] is alpha_payload

    refit_params = cd_cv_refit_get_params_result(self_params)
    assert refit_params == self_params
    assert refit_params is not self_params
    assert refit_params["alpha"] is alpha_payload

    model_payload = object()
    model_params = {"alpha": model_payload, "fit_intercept": True}
    copied_model_params = cd_cv_model_get_params_result(model_params)
    assert copied_model_params == model_params
    assert copied_model_params is not model_params
    assert copied_model_params["alpha"] is model_payload
    assert cd_cv_model_param_names(model_params) == {"alpha", "fit_intercept"}


def test_coordinate_descent_cv_estimator_params_callback_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_estimator_params_callback_shell import (
        cd_cv_get_estimator_result,
        cd_cv_model_get_params_result,
        cd_cv_model_param_names,
        cd_cv_path_get_params_result,
        cd_cv_refit_get_params_result,
    )

    with pytest.raises(ViolationError):
        cd_cv_get_estimator_result(None)

    with pytest.raises(ViolationError):
        cd_cv_path_get_params_result([("alpha", 1.0)])  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_cv_refit_get_params_result({1: "alpha"})  # type: ignore[dict-item]

    with pytest.raises(ViolationError):
        cd_cv_model_get_params_result({"alpha": 1.0, 2: "bad"})  # type: ignore[dict-item]

    with pytest.raises(ViolationError):
        cd_cv_model_param_names("alpha")  # type: ignore[arg-type]
