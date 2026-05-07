from __future__ import annotations

from numbers import Real

import pytest
from icontract import ViolationError
from sklearn.utils.validation import check_scalar


def test_coordinate_descent_cv_alpha_validation_callback_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_alpha_validation_callback_shell import (
        cd_cv_alpha_check_scalar_args,
        cd_cv_alpha_check_scalar_kwargs,
        cd_cv_alpha_check_scalar_result,
        cd_cv_user_alpha_validation_required,
    )

    assert callable(cd_cv_user_alpha_validation_required)
    assert callable(cd_cv_alpha_check_scalar_kwargs)
    assert callable(cd_cv_alpha_check_scalar_args)
    assert callable(cd_cv_alpha_check_scalar_result)


def test_coordinate_descent_cv_alpha_validation_callback_shell_matches_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_alpha_validation_callback_shell import (
        cd_cv_alpha_check_scalar_args,
        cd_cv_alpha_check_scalar_kwargs,
        cd_cv_alpha_check_scalar_result,
        cd_cv_user_alpha_validation_required,
    )

    assert cd_cv_user_alpha_validation_required(False) is True
    assert cd_cv_user_alpha_validation_required(True) is False

    kwargs = cd_cv_alpha_check_scalar_kwargs(Real)
    assert kwargs == {
        "target_type": Real,
        "min_val": 0.0,
        "include_boundaries": "left",
    }

    alpha = 0.25
    args = cd_cv_alpha_check_scalar_args(alpha, 3)
    assert args == (alpha, "alphas[3]")
    assert args[0] is alpha
    checked = check_scalar(*args, **kwargs)
    assert cd_cv_alpha_check_scalar_result(checked) == pytest.approx(alpha)


def test_coordinate_descent_cv_alpha_validation_callback_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_alpha_validation_callback_shell import (
        cd_cv_alpha_check_scalar_args,
        cd_cv_alpha_check_scalar_kwargs,
        cd_cv_user_alpha_validation_required,
    )

    with pytest.raises(ViolationError):
        cd_cv_user_alpha_validation_required(None)  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_cv_alpha_check_scalar_kwargs(None)

    with pytest.raises(ViolationError):
        cd_cv_alpha_check_scalar_args(0.1, -1)
