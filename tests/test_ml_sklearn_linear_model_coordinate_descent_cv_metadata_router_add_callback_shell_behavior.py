from __future__ import annotations

import pytest
from icontract import ViolationError


def test_coordinate_descent_cv_metadata_router_add_callback_shell_atom_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_metadata_router_add_callback_shell import (
        cd_cv_metadata_router_add_result,
    )

    assert callable(cd_cv_metadata_router_add_result)


def test_coordinate_descent_cv_metadata_router_add_callback_shell_preserves_result_identity() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_metadata_router_add_callback_shell import (
        cd_cv_metadata_router_add_result,
    )

    router_after_add = object()
    splitter = object()
    method_mapping = {"caller": "fit", "callee": "split"}

    result = cd_cv_metadata_router_add_result(router_after_add, splitter, method_mapping)

    assert result is router_after_add


def test_coordinate_descent_cv_metadata_router_add_callback_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_metadata_router_add_callback_shell import (
        cd_cv_metadata_router_add_result,
    )

    method_mapping = {"caller": "fit", "callee": "split"}

    with pytest.raises(ViolationError):
        cd_cv_metadata_router_add_result(None, object(), method_mapping)

    with pytest.raises(ViolationError):
        cd_cv_metadata_router_add_result(object(), None, method_mapping)

    with pytest.raises(ViolationError):
        cd_cv_metadata_router_add_result(object(), object(), {"caller": "fit", "callee": "score"})
