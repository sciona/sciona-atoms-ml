from __future__ import annotations

import numpy as np
import pytest
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF


def test_gpc_binary_log_marginal_likelihood_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_binary_log_marginal_likelihood_shell import (
        gpc_binary_log_marginal_likelihood_cached_result,
        gpc_binary_log_marginal_likelihood_kernel,
        gpc_binary_log_marginal_likelihood_require_theta_for_gradient,
        gpc_binary_log_marginal_likelihood_result,
        gpc_binary_log_marginal_likelihood_use_gradient_branch,
    )

    assert callable(gpc_binary_log_marginal_likelihood_require_theta_for_gradient)
    assert callable(gpc_binary_log_marginal_likelihood_cached_result)
    assert callable(gpc_binary_log_marginal_likelihood_kernel)
    assert callable(gpc_binary_log_marginal_likelihood_use_gradient_branch)
    assert callable(gpc_binary_log_marginal_likelihood_result)


def test_gpc_binary_log_marginal_likelihood_theta_guard_and_cached_result_match_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_binary_log_marginal_likelihood_shell import (
        gpc_binary_log_marginal_likelihood_cached_result,
        gpc_binary_log_marginal_likelihood_require_theta_for_gradient,
    )

    gpc_binary_log_marginal_likelihood_require_theta_for_gradient(False, False)
    gpc_binary_log_marginal_likelihood_require_theta_for_gradient(False, True)
    with pytest.raises(ValueError, match="Gradient can only be evaluated for theta!=None"):
        gpc_binary_log_marginal_likelihood_require_theta_for_gradient(True, True)

    assert gpc_binary_log_marginal_likelihood_cached_result(-2.5) == -2.5


def test_gpc_binary_log_marginal_likelihood_kernel_matches_clone_and_inplace_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_binary_log_marginal_likelihood_shell import (
        gpc_binary_log_marginal_likelihood_kernel,
    )

    theta = np.array([0.3, -0.1], dtype=np.float64)
    kernel = C(1.5) * RBF(0.8)

    cloned = gpc_binary_log_marginal_likelihood_kernel(kernel, theta, True)
    inplace = gpc_binary_log_marginal_likelihood_kernel(kernel, theta, False)

    assert cloned is not kernel
    assert np.allclose(np.asarray(cloned.theta, dtype=np.float64), theta)
    assert inplace is kernel
    assert np.allclose(np.asarray(kernel.theta, dtype=np.float64), theta)


def test_gpc_binary_log_marginal_likelihood_branch_and_result_match_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_binary_log_marginal_likelihood_shell import (
        gpc_binary_log_marginal_likelihood_result,
        gpc_binary_log_marginal_likelihood_use_gradient_branch,
    )

    gradient = np.array([1.0, -0.25], dtype=np.float64)

    assert gpc_binary_log_marginal_likelihood_use_gradient_branch(True) is True
    assert gpc_binary_log_marginal_likelihood_use_gradient_branch(False) is False
    assert gpc_binary_log_marginal_likelihood_result(-1.25, False) == -1.25
    observed = gpc_binary_log_marginal_likelihood_result(-1.25, True, gradient=gradient)
    assert observed[0] == -1.25
    assert np.array_equal(observed[1], gradient)
