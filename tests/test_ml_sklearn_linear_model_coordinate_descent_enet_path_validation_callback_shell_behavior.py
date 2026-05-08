from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_coordinate_descent_enet_path_validation_callback_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_enet_path_validation_callback_shell import (
        cd_enet_path_check_array_X_args,
        cd_enet_path_check_array_X_kwargs,
        cd_enet_path_check_array_Xy_args,
        cd_enet_path_check_array_Xy_kwargs,
        cd_enet_path_check_array_gram_args,
        cd_enet_path_check_array_gram_kwargs,
        cd_enet_path_check_array_y_args,
        cd_enet_path_check_array_y_kwargs,
    )

    assert callable(cd_enet_path_check_array_X_args)
    assert callable(cd_enet_path_check_array_X_kwargs)
    assert callable(cd_enet_path_check_array_y_args)
    assert callable(cd_enet_path_check_array_y_kwargs)
    assert callable(cd_enet_path_check_array_Xy_args)
    assert callable(cd_enet_path_check_array_Xy_kwargs)
    assert callable(cd_enet_path_check_array_gram_args)
    assert callable(cd_enet_path_check_array_gram_kwargs)


def test_coordinate_descent_enet_path_validation_callback_shell_matches_sklearn_setup() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_enet_path_validation_callback_shell import (
        cd_enet_path_check_array_X_args,
        cd_enet_path_check_array_X_kwargs,
        cd_enet_path_check_array_Xy_args,
        cd_enet_path_check_array_Xy_kwargs,
        cd_enet_path_check_array_gram_args,
        cd_enet_path_check_array_gram_kwargs,
        cd_enet_path_check_array_y_args,
        cd_enet_path_check_array_y_kwargs,
    )

    X = object()
    y = object()
    Xy = object()
    precompute = object()

    X_args = cd_enet_path_check_array_X_args(X)
    assert X_args == (X,)
    assert X_args[0] is X
    assert cd_enet_path_check_array_X_kwargs(True) == {
        "accept_sparse": "csc",
        "dtype": [np.float64, np.float32],
        "order": "F",
        "copy": True,
    }
    assert cd_enet_path_check_array_X_kwargs(False)["copy"] is False

    y_args = cd_enet_path_check_array_y_args(y)
    assert y_args == (y,)
    assert y_args[0] is y
    assert cd_enet_path_check_array_y_kwargs(np.float32) == {
        "accept_sparse": "csc",
        "dtype": np.float32,
        "order": "F",
        "copy": False,
        "ensure_2d": False,
    }

    Xy_args = cd_enet_path_check_array_Xy_args(Xy)
    assert Xy_args == (Xy,)
    assert Xy_args[0] is Xy
    assert cd_enet_path_check_array_Xy_kwargs(np.float64) == {
        "dtype": np.float64,
        "order": "C",
        "copy": False,
        "ensure_2d": False,
    }

    gram_args = cd_enet_path_check_array_gram_args(precompute)
    assert gram_args == (precompute,)
    assert gram_args[0] is precompute
    assert cd_enet_path_check_array_gram_kwargs(np.float64) == {
        "dtype": np.float64,
        "order": "C",
    }


def test_coordinate_descent_enet_path_validation_callback_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_enet_path_validation_callback_shell import (
        cd_enet_path_check_array_X_kwargs,
        cd_enet_path_check_array_Xy_kwargs,
        cd_enet_path_check_array_gram_kwargs,
        cd_enet_path_check_array_y_kwargs,
    )

    with pytest.raises(ViolationError):
        cd_enet_path_check_array_X_kwargs("yes")  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_enet_path_check_array_y_kwargs(object())

    with pytest.raises(ViolationError):
        cd_enet_path_check_array_Xy_kwargs(np.int64)

    with pytest.raises(ViolationError):
        cd_enet_path_check_array_gram_kwargs(None)
