from __future__ import annotations

import pytest
from icontract import ViolationError


def test_coordinate_descent_cv_tags_super_callback_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_tags_super_callback_shell import (
        cd_cv_tags_return,
        cd_cv_tags_super_result,
    )

    assert callable(cd_cv_tags_super_result)
    assert callable(cd_cv_tags_return)


def test_coordinate_descent_cv_tags_super_callback_shell_preserves_tags_identity() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_tags_super_callback_shell import (
        cd_cv_tags_return,
        cd_cv_tags_super_result,
    )

    tags = object()

    assert cd_cv_tags_super_result(tags) is tags
    assert cd_cv_tags_return(tags) is tags


def test_coordinate_descent_cv_tags_super_callback_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_tags_super_callback_shell import (
        cd_cv_tags_return,
        cd_cv_tags_super_result,
    )

    with pytest.raises(ViolationError):
        cd_cv_tags_super_result(None)

    with pytest.raises(ViolationError):
        cd_cv_tags_return(None)
