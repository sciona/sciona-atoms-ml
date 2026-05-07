from __future__ import annotations

import pytest
from icontract import ViolationError


def test_coordinate_descent_cv_validation_callback_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_validation_callback_shell import (
        cd_cv_check_consistent_length_args,
        cd_cv_validate_data_args,
        cd_cv_validate_data_kwargs,
        cd_cv_validated_x,
        cd_cv_validated_y,
    )

    assert callable(cd_cv_validate_data_args)
    assert callable(cd_cv_validate_data_kwargs)
    assert callable(cd_cv_validated_x)
    assert callable(cd_cv_validated_y)
    assert callable(cd_cv_check_consistent_length_args)


def test_coordinate_descent_cv_validation_callback_shell_matches_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_validation_callback_shell import (
        cd_cv_check_consistent_length_args,
        cd_cv_validate_data_args,
        cd_cv_validate_data_kwargs,
        cd_cv_validated_x,
        cd_cv_validated_y,
    )

    estimator = object()
    X = object()
    y = object()
    args = cd_cv_validate_data_args(estimator, X, y)
    assert args == (estimator, X, y)
    assert args[0] is estimator
    assert args[1] is X
    assert args[2] is y

    check_x_params = {"accept_sparse": "csc", "copy": False}
    check_y_params = {"ensure_2d": False, "copy": False}
    kwargs = cd_cv_validate_data_kwargs(check_x_params, check_y_params)
    assert kwargs == {"validate_separately": (check_x_params, check_y_params)}

    validated_x = object()
    validated_y = object()
    validated_pair = (validated_x, validated_y)
    assert cd_cv_validated_x(validated_pair) is validated_x
    assert cd_cv_validated_y(validated_pair) is validated_y

    consistent_args = cd_cv_check_consistent_length_args(validated_x, validated_y)
    assert consistent_args == (validated_x, validated_y)
    assert consistent_args[0] is validated_x
    assert consistent_args[1] is validated_y


def test_coordinate_descent_cv_validation_callback_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_validation_callback_shell import (
        cd_cv_validate_data_kwargs,
        cd_cv_validated_x,
        cd_cv_validated_y,
    )

    with pytest.raises(ViolationError):
        cd_cv_validate_data_kwargs([], {})  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_cv_validated_x((object(), object(), object()))  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_cv_validated_y(object())  # type: ignore[arg-type]
