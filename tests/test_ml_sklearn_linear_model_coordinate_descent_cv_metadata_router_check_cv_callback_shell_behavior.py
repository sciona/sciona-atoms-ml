from __future__ import annotations

import pytest
from icontract import ViolationError


def test_coordinate_descent_cv_metadata_router_check_cv_callback_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_metadata_router_check_cv_callback_shell import (
        cd_cv_metadata_router_check_cv_args,
        cd_cv_metadata_router_checked_splitter_result,
    )

    assert callable(cd_cv_metadata_router_check_cv_args)
    assert callable(cd_cv_metadata_router_checked_splitter_result)


def test_coordinate_descent_cv_metadata_router_check_cv_callback_shell_preserves_cv_arg_identity() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_metadata_router_check_cv_callback_shell import (
        cd_cv_metadata_router_check_cv_args,
    )

    cv = 5

    result = cd_cv_metadata_router_check_cv_args(cv)

    assert result == (cv,)
    assert result[0] is cv


def test_coordinate_descent_cv_metadata_router_check_cv_callback_shell_preserves_splitter_result_identity() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_metadata_router_check_cv_callback_shell import (
        cd_cv_metadata_router_checked_splitter_result,
    )

    checked_splitter = object()

    result = cd_cv_metadata_router_checked_splitter_result(checked_splitter)

    assert result is checked_splitter


def test_coordinate_descent_cv_metadata_router_check_cv_callback_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_metadata_router_check_cv_callback_shell import (
        cd_cv_metadata_router_check_cv_args,
        cd_cv_metadata_router_checked_splitter_result,
    )

    with pytest.raises(ViolationError):
        cd_cv_metadata_router_check_cv_args("not-a-cv")

    with pytest.raises(ViolationError):
        cd_cv_metadata_router_checked_splitter_result(None)
