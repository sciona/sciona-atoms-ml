from __future__ import annotations

import pytest
from icontract import ViolationError


def test_coordinate_descent_estimator_intercept_callback_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_estimator_intercept_callback_shell import (
        cd_estimator_fit_return_self,
        cd_estimator_set_intercept_args,
    )

    assert callable(cd_estimator_set_intercept_args)
    assert callable(cd_estimator_fit_return_self)


def test_coordinate_descent_estimator_intercept_callback_shell_matches_sklearn_tail() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_estimator_intercept_callback_shell import (
        cd_estimator_fit_return_self,
        cd_estimator_set_intercept_args,
    )

    X_offset = object()
    y_offset = object()
    X_scale = object()

    args = cd_estimator_set_intercept_args(X_offset, y_offset, X_scale)
    assert args == (X_offset, y_offset, X_scale)
    assert args[0] is X_offset
    assert args[1] is y_offset
    assert args[2] is X_scale

    token = object()
    assert cd_estimator_fit_return_self(token) is token


def test_coordinate_descent_estimator_intercept_callback_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_estimator_intercept_callback_shell import (
        cd_estimator_fit_return_self,
        cd_estimator_set_intercept_args,
    )

    with pytest.raises(ViolationError):
        cd_estimator_set_intercept_args(None, object(), object())

    with pytest.raises(ViolationError):
        cd_estimator_set_intercept_args(object(), None, object())

    with pytest.raises(ViolationError):
        cd_estimator_set_intercept_args(object(), object(), None)

    with pytest.raises(ViolationError):
        cd_estimator_fit_return_self(None)
