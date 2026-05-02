from __future__ import annotations

import numpy as np


def test_gp_regression_fit_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_fit_shell import (
        gp_fit_dtype_name,
        gp_fit_stored_train_inputs,
        gp_fit_stored_train_targets,
        gp_fit_use_optimizer_branch,
        gp_fit_validate_ensure_2d,
    )

    assert callable(gp_fit_dtype_name)
    assert callable(gp_fit_validate_ensure_2d)
    assert callable(gp_fit_use_optimizer_branch)
    assert callable(gp_fit_stored_train_inputs)
    assert callable(gp_fit_stored_train_targets)


def test_gp_fit_validation_mode_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_fit_shell import (
        gp_fit_dtype_name,
        gp_fit_validate_ensure_2d,
    )

    assert gp_fit_dtype_name(True) == "numeric"
    assert gp_fit_validate_ensure_2d(True) is True

    assert gp_fit_dtype_name(False) is None
    assert gp_fit_validate_ensure_2d(False) is False


def test_gp_fit_use_optimizer_branch_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_fit_shell import (
        gp_fit_use_optimizer_branch,
    )

    assert gp_fit_use_optimizer_branch(True, 1) is True
    assert gp_fit_use_optimizer_branch(True, 0) is False
    assert gp_fit_use_optimizer_branch(False, 3) is False


def test_gp_fit_stored_train_inputs_respects_copy_policy() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_fit_shell import (
        gp_fit_stored_train_inputs,
    )

    X = np.array([[1.0, 2.0], [3.0, 4.0]])

    shared = gp_fit_stored_train_inputs(X, False)
    copied = gp_fit_stored_train_inputs(X, True)

    assert shared is X
    assert copied is not X
    assert np.array_equal(shared, X)
    assert np.array_equal(copied, X)


def test_gp_fit_stored_train_targets_respects_copy_policy() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_fit_shell import (
        gp_fit_stored_train_targets,
    )

    y = np.array([1.0, -1.0, 2.0])

    shared = gp_fit_stored_train_targets(y, False)
    copied = gp_fit_stored_train_targets(y, True)

    assert shared is y
    assert copied is not y
    assert np.array_equal(shared, y)
    assert np.array_equal(copied, y)
