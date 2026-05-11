from __future__ import annotations

import pytest
from icontract import ViolationError


def test_coordinate_descent_cv_metadata_router_method_mapping_callback_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_metadata_router_method_mapping_callback_shell import (
        cd_cv_metadata_router_method_mapping_add_kwargs,
        cd_cv_metadata_router_method_mapping_result,
    )

    assert callable(cd_cv_metadata_router_method_mapping_add_kwargs)
    assert callable(cd_cv_metadata_router_method_mapping_result)


def test_coordinate_descent_cv_metadata_router_method_mapping_callback_shell_matches_fit_split_mapping() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_metadata_router_method_mapping_callback_shell import (
        cd_cv_metadata_router_method_mapping_add_kwargs,
        cd_cv_metadata_router_method_mapping_result,
    )

    kwargs = cd_cv_metadata_router_method_mapping_add_kwargs("fit", "split")

    assert kwargs == {"caller": "fit", "callee": "split"}
    assert cd_cv_metadata_router_method_mapping_result(kwargs) is kwargs


def test_coordinate_descent_cv_metadata_router_method_mapping_callback_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_metadata_router_method_mapping_callback_shell import (
        cd_cv_metadata_router_method_mapping_add_kwargs,
        cd_cv_metadata_router_method_mapping_result,
    )

    with pytest.raises(ViolationError):
        cd_cv_metadata_router_method_mapping_add_kwargs("predict", "split")

    with pytest.raises(ViolationError):
        cd_cv_metadata_router_method_mapping_add_kwargs("fit", "score")

    with pytest.raises(ViolationError):
        cd_cv_metadata_router_method_mapping_result({"caller": "fit", "callee": "score"})
