from __future__ import annotations

import numpy as np


def test_gp_regression_constrained_optimization_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_constrained_optimization_shell import (
        gp_constrained_optimization_result,
        gp_constrained_optimization_unknown_optimizer_message,
        gp_constrained_optimization_use_callable,
        gp_constrained_optimization_use_lbfgsb,
    )

    assert callable(gp_constrained_optimization_use_lbfgsb)
    assert callable(gp_constrained_optimization_use_callable)
    assert callable(gp_constrained_optimization_unknown_optimizer_message)
    assert callable(gp_constrained_optimization_result)


def test_gp_constrained_optimization_branch_predicates_match_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_constrained_optimization_shell import (
        gp_constrained_optimization_use_callable,
        gp_constrained_optimization_use_lbfgsb,
    )

    assert gp_constrained_optimization_use_lbfgsb("fmin_l_bfgs_b") is True
    assert gp_constrained_optimization_use_lbfgsb("custom") is False

    assert gp_constrained_optimization_use_callable(True, "custom") is True
    assert gp_constrained_optimization_use_callable(False, "custom") is False
    assert gp_constrained_optimization_use_callable(True, "fmin_l_bfgs_b") is False


def test_gp_constrained_optimization_unknown_optimizer_message_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_constrained_optimization_shell import (
        gp_constrained_optimization_unknown_optimizer_message,
    )

    assert gp_constrained_optimization_unknown_optimizer_message("custom") == "Unknown optimizer custom."


def test_gp_constrained_optimization_result_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_constrained_optimization_shell import (
        gp_constrained_optimization_result,
    )

    theta = np.array([0.5, 1.25], dtype=np.float64)
    func_min = -3.0
    result = gp_constrained_optimization_result(theta, func_min)

    assert np.array_equal(result[0], theta)
    assert result[1] == func_min
