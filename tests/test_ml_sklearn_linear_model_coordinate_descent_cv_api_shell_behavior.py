from __future__ import annotations

import pytest
from icontract import ViolationError


def test_coordinate_descent_cv_api_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_api_shell import (
        cd_cv_metadata_router_spec,
        cd_cv_multitask_bool,
        cd_cv_sparse_input_tag,
        cd_cv_target_multi_output_tag,
        cd_cv_target_single_output_tag,
    )

    assert callable(cd_cv_metadata_router_spec)
    assert callable(cd_cv_multitask_bool)
    assert callable(cd_cv_sparse_input_tag)
    assert callable(cd_cv_target_multi_output_tag)
    assert callable(cd_cv_target_single_output_tag)


def test_coordinate_descent_cv_api_shell_matches_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_api_shell import (
        cd_cv_metadata_router_spec,
        cd_cv_multitask_bool,
        cd_cv_sparse_input_tag,
        cd_cv_target_multi_output_tag,
        cd_cv_target_single_output_tag,
    )

    assert cd_cv_metadata_router_spec("ElasticNetCV") == {
        "owner": "ElasticNetCV",
        "caller": "fit",
        "callee": "split",
    }
    multitask = cd_cv_multitask_bool(True)
    assert multitask is True
    assert cd_cv_sparse_input_tag(multitask) is False
    assert cd_cv_target_multi_output_tag(multitask) is True
    assert cd_cv_target_single_output_tag(multitask) is False


def test_coordinate_descent_cv_api_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_api_shell import (
        cd_cv_metadata_router_spec,
        cd_cv_multitask_bool,
    )

    with pytest.raises(ViolationError):
        cd_cv_metadata_router_spec("")

    with pytest.raises(ViolationError):
        cd_cv_multitask_bool(1)
