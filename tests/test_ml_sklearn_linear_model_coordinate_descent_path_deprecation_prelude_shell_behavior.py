from __future__ import annotations

import pytest
from icontract import ViolationError


def test_coordinate_descent_path_deprecation_prelude_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_deprecation_prelude_shell import (
        cd_path_alphas_none_warning_message,
        cd_path_alphas_none_warning_required,
        cd_path_default_n_alphas_resolution,
        cd_path_effective_alphas_resolution,
        cd_path_n_alphas_warning_message,
        cd_path_n_alphas_warning_required,
    )

    assert callable(cd_path_default_n_alphas_resolution)
    assert callable(cd_path_n_alphas_warning_required)
    assert callable(cd_path_n_alphas_warning_message)
    assert callable(cd_path_alphas_none_warning_required)
    assert callable(cd_path_alphas_none_warning_message)
    assert callable(cd_path_effective_alphas_resolution)


def test_coordinate_descent_path_deprecation_prelude_shell_matches_path_branches() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_deprecation_prelude_shell import (
        cd_path_alphas_none_warning_message,
        cd_path_alphas_none_warning_required,
        cd_path_default_n_alphas_resolution,
        cd_path_effective_alphas_resolution,
        cd_path_n_alphas_warning_message,
        cd_path_n_alphas_warning_required,
    )

    explicit_alphas = [3.0, 1.0, 0.3]

    assert cd_path_default_n_alphas_resolution("deprecated") == 100
    assert cd_path_default_n_alphas_resolution(25) == 25
    assert cd_path_n_alphas_warning_required("deprecated") is False
    assert cd_path_n_alphas_warning_required(25) is True
    assert cd_path_alphas_none_warning_required(None) is True
    assert cd_path_alphas_none_warning_required("warn") is False

    assert cd_path_effective_alphas_resolution("deprecated", "warn") == {
        "effective_alphas": 100,
        "warn_n_alphas": False,
        "warn_alphas_none": False,
    }
    assert cd_path_effective_alphas_resolution(25, "warn") == {
        "effective_alphas": 25,
        "warn_n_alphas": True,
        "warn_alphas_none": False,
    }
    assert cd_path_effective_alphas_resolution("deprecated", None) == {
        "effective_alphas": 100,
        "warn_n_alphas": False,
        "warn_alphas_none": True,
    }
    assert cd_path_effective_alphas_resolution("deprecated", explicit_alphas) == {
        "effective_alphas": explicit_alphas,
        "warn_n_alphas": False,
        "warn_alphas_none": False,
    }

    assert "'n_alphas' was deprecated in 1.9" in cd_path_n_alphas_warning_message("lasso_path")
    assert "will be removed in 1.11" in cd_path_n_alphas_warning_message("enet_path")
    assert "Set 'alphas=100'" in cd_path_alphas_none_warning_message("lasso_path")
    assert "will be removed in 1.11" in cd_path_alphas_none_warning_message("enet_path")


def test_coordinate_descent_path_deprecation_prelude_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_deprecation_prelude_shell import (
        cd_path_alphas_none_warning_message,
        cd_path_default_n_alphas_resolution,
        cd_path_effective_alphas_resolution,
        cd_path_n_alphas_warning_message,
    )

    with pytest.raises(ViolationError):
        cd_path_default_n_alphas_resolution(0)

    with pytest.raises(ViolationError):
        cd_path_n_alphas_warning_message("ridge_path")

    with pytest.raises(ViolationError):
        cd_path_alphas_none_warning_message("ridge_path")

    with pytest.raises(ViolationError):
        cd_path_effective_alphas_resolution("deprecated", "bad")
