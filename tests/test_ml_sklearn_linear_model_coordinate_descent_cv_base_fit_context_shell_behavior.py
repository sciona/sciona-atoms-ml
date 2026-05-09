from __future__ import annotations

import pytest
from icontract import ViolationError
from sklearn.linear_model._coordinate_descent import LinearModelCV


def test_coordinate_descent_cv_base_fit_context_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_base_fit_context_shell import (
        cd_cv_base_fit_context_kwargs,
        cd_cv_base_fit_context_method_name,
    )

    assert callable(cd_cv_base_fit_context_kwargs)
    assert callable(cd_cv_base_fit_context_method_name)


def test_coordinate_descent_cv_base_fit_context_shell_matches_sklearn_decorated_method() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_base_fit_context_shell import (
        cd_cv_base_fit_context_kwargs,
        cd_cv_base_fit_context_method_name,
    )

    assert LinearModelCV.fit.__name__ == "fit"
    assert cd_cv_base_fit_context_method_name(LinearModelCV.fit.__name__) == "fit"
    assert cd_cv_base_fit_context_kwargs("fit") == {
        "prefer_skip_nested_validation": True,
    }


def test_coordinate_descent_cv_base_fit_context_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_base_fit_context_shell import (
        cd_cv_base_fit_context_kwargs,
        cd_cv_base_fit_context_method_name,
    )

    with pytest.raises(ViolationError):
        cd_cv_base_fit_context_kwargs("predict")

    with pytest.raises(ViolationError):
        cd_cv_base_fit_context_method_name("transform")
