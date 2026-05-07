from __future__ import annotations

import pytest
from icontract import ViolationError


def test_coordinate_descent_cv_target_callback_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_target_callback_shell import (
        cd_cv_check_sample_weight_args,
        cd_cv_check_sample_weight_kwargs,
        cd_cv_checked_sample_weight,
        cd_cv_column_or_1d_args,
        cd_cv_column_or_1d_result,
        cd_cv_is_multitask_result,
    )

    assert callable(cd_cv_is_multitask_result)
    assert callable(cd_cv_column_or_1d_args)
    assert callable(cd_cv_column_or_1d_result)
    assert callable(cd_cv_check_sample_weight_args)
    assert callable(cd_cv_check_sample_weight_kwargs)
    assert callable(cd_cv_checked_sample_weight)


def test_coordinate_descent_cv_target_callback_shell_matches_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_target_callback_shell import (
        cd_cv_check_sample_weight_args,
        cd_cv_check_sample_weight_kwargs,
        cd_cv_checked_sample_weight,
        cd_cv_column_or_1d_args,
        cd_cv_column_or_1d_result,
        cd_cv_is_multitask_result,
    )

    assert cd_cv_is_multitask_result(False) is False

    y = object()
    column_args = cd_cv_column_or_1d_args(y, False)
    assert column_args == (y,)
    assert column_args[0] is y
    normalized_y = object()
    assert cd_cv_column_or_1d_result(normalized_y) is normalized_y

    sample_weight = object()
    X = object()
    weight_args = cd_cv_check_sample_weight_args(sample_weight, X)
    assert weight_args == (sample_weight, X)
    assert weight_args[0] is sample_weight
    assert weight_args[1] is X
    assert cd_cv_check_sample_weight_kwargs("float64") == {"dtype": "float64"}
    checked = object()
    assert cd_cv_checked_sample_weight(checked) is checked


def test_coordinate_descent_cv_target_callback_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_target_callback_shell import (
        cd_cv_check_sample_weight_args,
        cd_cv_column_or_1d_args,
        cd_cv_is_multitask_result,
    )

    with pytest.raises(ViolationError):
        cd_cv_is_multitask_result("no")  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_cv_column_or_1d_args(object(), True)

    with pytest.raises(ViolationError):
        cd_cv_check_sample_weight_args(None, object())
