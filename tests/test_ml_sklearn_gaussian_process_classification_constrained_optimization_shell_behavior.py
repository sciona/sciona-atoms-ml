from __future__ import annotations

import numpy as np


def test_gpc_classification_constrained_optimization_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_constrained_optimization_shell import (
        gpc_constrained_optimization_result,
        gpc_constrained_optimization_unknown_optimizer_message,
        gpc_constrained_optimization_use_callable,
        gpc_constrained_optimization_use_lbfgsb,
    )

    assert callable(gpc_constrained_optimization_use_lbfgsb)
    assert callable(gpc_constrained_optimization_use_callable)
    assert callable(gpc_constrained_optimization_unknown_optimizer_message)
    assert callable(gpc_constrained_optimization_result)


def test_gpc_constrained_optimization_branch_predicates_match_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_constrained_optimization_shell import (
        gpc_constrained_optimization_use_callable,
        gpc_constrained_optimization_use_lbfgsb,
    )

    assert gpc_constrained_optimization_use_lbfgsb("fmin_l_bfgs_b") is True
    assert gpc_constrained_optimization_use_lbfgsb("custom") is False

    assert gpc_constrained_optimization_use_callable(True, "custom") is True
    assert gpc_constrained_optimization_use_callable(False, "custom") is False
    assert gpc_constrained_optimization_use_callable(True, "fmin_l_bfgs_b") is False


def test_gpc_constrained_optimization_unknown_optimizer_message_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_constrained_optimization_shell import (
        gpc_constrained_optimization_unknown_optimizer_message,
    )

    assert gpc_constrained_optimization_unknown_optimizer_message("custom") == "Unknown optimizer custom."


def test_gpc_constrained_optimization_result_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_constrained_optimization_shell import (
        gpc_constrained_optimization_result,
    )

    theta = np.array([0.5, 1.25], dtype=np.float64)
    func_min = -3.0
    result = gpc_constrained_optimization_result(theta, func_min)

    assert np.array_equal(result[0], theta)
    assert result[1] == func_min
