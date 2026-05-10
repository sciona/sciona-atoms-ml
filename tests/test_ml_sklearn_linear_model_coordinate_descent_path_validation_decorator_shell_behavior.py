from __future__ import annotations

import pytest
from icontract import ViolationError
from sklearn.linear_model._coordinate_descent import enet_path, lasso_path


def test_coordinate_descent_path_validation_decorator_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_validation_decorator_shell import (
        cd_enet_path_validation_param_names,
        cd_lasso_path_validation_param_names,
        cd_path_validation_param_descriptors,
        cd_path_validation_prefers_skip_nested,
    )

    assert callable(cd_lasso_path_validation_param_names)
    assert callable(cd_enet_path_validation_param_names)
    assert callable(cd_path_validation_param_descriptors)
    assert callable(cd_path_validation_prefers_skip_nested)


def test_coordinate_descent_path_validation_decorator_shell_names_match_sklearn_decorators() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_validation_decorator_shell import (
        cd_enet_path_validation_param_names,
        cd_lasso_path_validation_param_names,
    )

    assert cd_lasso_path_validation_param_names("lasso_path") == tuple(
        lasso_path._skl_parameter_constraints
    )
    assert cd_enet_path_validation_param_names("enet_path") == tuple(
        enet_path._skl_parameter_constraints
    )


def test_coordinate_descent_path_validation_decorator_shell_descriptors_and_flag() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_validation_decorator_shell import (
        cd_path_validation_param_descriptors,
        cd_path_validation_prefers_skip_nested,
    )

    lasso = cd_path_validation_param_descriptors("lasso_path")
    enet = cd_path_validation_param_descriptors("enet_path")

    assert tuple(lasso) == tuple(lasso_path._skl_parameter_constraints)
    assert tuple(enet) == tuple(enet_path._skl_parameter_constraints)
    assert "l1_ratio" not in lasso
    assert "check_input" not in lasso
    assert enet["l1_ratio"] == (("interval", "Real", 0.0, 1.0, "both"),)
    assert enet["check_input"] == ("boolean",)
    assert lasso["precompute"] == (("str_options", ("auto",)), "boolean", "array-like")
    assert cd_path_validation_prefers_skip_nested("lasso_path") is True
    assert cd_path_validation_prefers_skip_nested("enet_path") is True


def test_coordinate_descent_path_validation_decorator_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_validation_decorator_shell import (
        cd_enet_path_validation_param_names,
        cd_lasso_path_validation_param_names,
        cd_path_validation_param_descriptors,
        cd_path_validation_prefers_skip_nested,
    )

    with pytest.raises(ViolationError):
        cd_lasso_path_validation_param_names("enet_path")

    with pytest.raises(ViolationError):
        cd_enet_path_validation_param_names("lasso_path")

    with pytest.raises(ViolationError):
        cd_path_validation_param_descriptors("elastic_net_cv")

    with pytest.raises(ViolationError):
        cd_path_validation_prefers_skip_nested("elastic_net_cv")
