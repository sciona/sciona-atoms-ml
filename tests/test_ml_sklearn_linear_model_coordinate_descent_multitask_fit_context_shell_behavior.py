from __future__ import annotations

import pytest
from icontract import ViolationError
from sklearn.linear_model import MultiTaskElasticNet


def test_coordinate_descent_multitask_fit_context_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_multitask_fit_context_shell import (
        cd_multitask_fit_context_kwargs,
        cd_multitask_fit_context_method_name,
    )

    assert callable(cd_multitask_fit_context_kwargs)
    assert callable(cd_multitask_fit_context_method_name)


def test_coordinate_descent_multitask_fit_context_shell_matches_sklearn_decorated_method() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_multitask_fit_context_shell import (
        cd_multitask_fit_context_kwargs,
        cd_multitask_fit_context_method_name,
    )

    assert MultiTaskElasticNet.fit.__name__ == "fit"
    assert (
        cd_multitask_fit_context_method_name(
            "MultiTaskElasticNet", MultiTaskElasticNet.fit.__name__
        )
        == "fit"
    )
    assert cd_multitask_fit_context_kwargs("MultiTaskElasticNet") == {
        "prefer_skip_nested_validation": True,
    }


def test_coordinate_descent_multitask_fit_context_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_multitask_fit_context_shell import (
        cd_multitask_fit_context_kwargs,
        cd_multitask_fit_context_method_name,
    )

    with pytest.raises(ViolationError):
        cd_multitask_fit_context_kwargs("LinearModelCV")

    with pytest.raises(ViolationError):
        cd_multitask_fit_context_method_name("LinearModelCV", "fit")

    with pytest.raises(ViolationError):
        cd_multitask_fit_context_method_name("MultiTaskElasticNet", "predict")
