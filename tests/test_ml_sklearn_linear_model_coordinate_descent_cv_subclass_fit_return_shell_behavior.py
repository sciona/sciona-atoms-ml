from __future__ import annotations

import pytest
from icontract import ViolationError


CV_KINDS = (
    "lasso_cv",
    "elastic_net_cv",
    "multi_task_lasso_cv",
    "multi_task_elastic_net_cv",
)


def test_coordinate_descent_cv_subclass_fit_return_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_subclass_fit_return_shell import (
        cd_cv_subclass_fit_returns_super_result,
        cd_cv_subclass_return_passthrough_required,
    )

    assert callable(cd_cv_subclass_return_passthrough_required)
    assert callable(cd_cv_subclass_fit_returns_super_result)


def test_coordinate_descent_cv_subclass_fit_return_shell_matches_sklearn_wrappers() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_subclass_fit_return_shell import (
        cd_cv_subclass_fit_returns_super_result,
        cd_cv_subclass_return_passthrough_required,
    )

    fitted = object()

    for cv_kind in CV_KINDS:
        assert cd_cv_subclass_return_passthrough_required(cv_kind) is True
        assert cd_cv_subclass_fit_returns_super_result(cv_kind, fitted) is fitted


def test_coordinate_descent_cv_subclass_fit_return_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_subclass_fit_return_shell import (
        cd_cv_subclass_fit_returns_super_result,
        cd_cv_subclass_return_passthrough_required,
    )

    with pytest.raises(ViolationError):
        cd_cv_subclass_return_passthrough_required("ridge_cv")

    with pytest.raises(ViolationError):
        cd_cv_subclass_fit_returns_super_result("ridge_cv", object())

    with pytest.raises(ViolationError):
        cd_cv_subclass_fit_returns_super_result("lasso_cv", None)
