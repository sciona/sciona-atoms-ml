from __future__ import annotations

import pytest
from icontract import ViolationError


def test_coordinate_descent_cv_parallel_callback_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_parallel_callback_shell import (
        cd_cv_parallel_kwargs,
        cd_cv_parallel_mse_paths,
    )

    assert callable(cd_cv_parallel_kwargs)
    assert callable(cd_cv_parallel_mse_paths)


def test_coordinate_descent_cv_parallel_callback_shell_matches_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_parallel_callback_shell import (
        cd_cv_parallel_kwargs,
        cd_cv_parallel_mse_paths,
    )

    assert cd_cv_parallel_kwargs(None, 2) == {
        "n_jobs": None,
        "verbose": 2,
        "prefer": "threads",
    }
    mse_paths = [0.3, 0.2, 0.1]
    assert cd_cv_parallel_mse_paths(mse_paths) == mse_paths


def test_coordinate_descent_cv_parallel_callback_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_parallel_callback_shell import (
        cd_cv_parallel_kwargs,
        cd_cv_parallel_mse_paths,
    )

    with pytest.raises(ViolationError):
        cd_cv_parallel_kwargs(1, "loud")

    with pytest.raises(ViolationError):
        cd_cv_parallel_mse_paths(object())
