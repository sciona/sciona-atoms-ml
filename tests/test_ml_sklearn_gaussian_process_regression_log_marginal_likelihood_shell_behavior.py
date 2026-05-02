from __future__ import annotations

import numpy as np
import pytest
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF


def test_gp_regression_log_marginal_likelihood_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_log_marginal_likelihood_shell import (
        gp_log_marginal_likelihood_cached_result,
        gp_log_marginal_likelihood_cholesky_failure_result,
        gp_log_marginal_likelihood_kernel,
        gp_log_marginal_likelihood_require_theta_for_gradient,
        gp_log_marginal_likelihood_train_targets,
    )

    assert callable(gp_log_marginal_likelihood_require_theta_for_gradient)
    assert callable(gp_log_marginal_likelihood_cached_result)
    assert callable(gp_log_marginal_likelihood_kernel)
    assert callable(gp_log_marginal_likelihood_cholesky_failure_result)
    assert callable(gp_log_marginal_likelihood_train_targets)


def test_gp_log_marginal_likelihood_require_theta_for_gradient_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_log_marginal_likelihood_shell import (
        gp_log_marginal_likelihood_require_theta_for_gradient,
    )

    gp_log_marginal_likelihood_require_theta_for_gradient(False, False)
    gp_log_marginal_likelihood_require_theta_for_gradient(False, True)
    gp_log_marginal_likelihood_require_theta_for_gradient(True, False)

    with pytest.raises(ValueError, match="Gradient can only be evaluated for theta!=None"):
        gp_log_marginal_likelihood_require_theta_for_gradient(True, True)


def test_gp_log_marginal_likelihood_cached_result_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_log_marginal_likelihood_shell import (
        gp_log_marginal_likelihood_cached_result,
    )

    assert gp_log_marginal_likelihood_cached_result(-3.25) == -3.25


def test_gp_log_marginal_likelihood_kernel_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_log_marginal_likelihood_shell import (
        gp_log_marginal_likelihood_kernel,
    )

    kernel = C(1.0) * RBF(2.0)
    theta = np.array(kernel.theta, dtype=np.float64)

    cloned = gp_log_marginal_likelihood_kernel(kernel, theta, True)
    assert cloned is not kernel
    assert np.array_equal(cloned.theta, theta)

    inplace = gp_log_marginal_likelihood_kernel(kernel, theta, False)
    assert inplace is kernel
    assert np.array_equal(inplace.theta, theta)


def test_gp_log_marginal_likelihood_cholesky_failure_result_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_log_marginal_likelihood_shell import (
        gp_log_marginal_likelihood_cholesky_failure_result,
    )

    theta = np.array([0.5, 1.5], dtype=np.float64)
    assert gp_log_marginal_likelihood_cholesky_failure_result(theta, False) == float("-inf")

    failure = gp_log_marginal_likelihood_cholesky_failure_result(theta, True)
    assert failure[0] == float("-inf")
    assert np.array_equal(failure[1], np.zeros_like(theta))


def test_gp_log_marginal_likelihood_train_targets_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_log_marginal_likelihood_shell import (
        gp_log_marginal_likelihood_train_targets,
    )

    y1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    result1 = gp_log_marginal_likelihood_train_targets(y1)
    assert result1.shape == (3, 1)
    assert np.array_equal(result1[:, 0], y1)

    y2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    result2 = gp_log_marginal_likelihood_train_targets(y2)
    assert np.array_equal(result2, y2)
