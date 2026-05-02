from __future__ import annotations

import numpy as np
import pytest


def test_gpc_log_marginal_likelihood_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_log_marginal_likelihood_shell import (
        gpc_log_marginal_likelihood_cached_result,
        gpc_log_marginal_likelihood_mean,
        gpc_log_marginal_likelihood_require_no_multiclass_gradient,
        gpc_log_marginal_likelihood_require_theta_for_gradient,
        gpc_log_marginal_likelihood_theta_shape_message,
        gpc_log_marginal_likelihood_theta_slice,
        gpc_log_marginal_likelihood_use_binary_branch,
        gpc_log_marginal_likelihood_use_compound_theta,
        gpc_log_marginal_likelihood_use_shared_theta,
    )

    assert callable(gpc_log_marginal_likelihood_require_theta_for_gradient)
    assert callable(gpc_log_marginal_likelihood_cached_result)
    assert callable(gpc_log_marginal_likelihood_require_no_multiclass_gradient)
    assert callable(gpc_log_marginal_likelihood_use_binary_branch)
    assert callable(gpc_log_marginal_likelihood_use_shared_theta)
    assert callable(gpc_log_marginal_likelihood_use_compound_theta)
    assert callable(gpc_log_marginal_likelihood_theta_slice)
    assert callable(gpc_log_marginal_likelihood_mean)
    assert callable(gpc_log_marginal_likelihood_theta_shape_message)


def test_gpc_log_marginal_likelihood_guards_match_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_log_marginal_likelihood_shell import (
        gpc_log_marginal_likelihood_cached_result,
        gpc_log_marginal_likelihood_require_no_multiclass_gradient,
        gpc_log_marginal_likelihood_require_theta_for_gradient,
        gpc_log_marginal_likelihood_use_binary_branch,
    )

    gpc_log_marginal_likelihood_require_theta_for_gradient(False, False)
    gpc_log_marginal_likelihood_require_theta_for_gradient(False, True)
    with pytest.raises(ValueError, match="Gradient can only be evaluated for theta!=None"):
        gpc_log_marginal_likelihood_require_theta_for_gradient(True, True)

    gpc_log_marginal_likelihood_require_no_multiclass_gradient(2, True)
    with pytest.raises(NotImplementedError, match="Gradient of log-marginal-likelihood not implemented for multi-class GPC."):
        gpc_log_marginal_likelihood_require_no_multiclass_gradient(3, True)

    assert gpc_log_marginal_likelihood_cached_result(-2.5) == -2.5
    assert gpc_log_marginal_likelihood_use_binary_branch(2) is True
    assert gpc_log_marginal_likelihood_use_binary_branch(3) is False


def test_gpc_log_marginal_likelihood_theta_routing_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_log_marginal_likelihood_shell import (
        gpc_log_marginal_likelihood_mean,
        gpc_log_marginal_likelihood_theta_shape_message,
        gpc_log_marginal_likelihood_theta_slice,
        gpc_log_marginal_likelihood_use_compound_theta,
        gpc_log_marginal_likelihood_use_shared_theta,
    )

    shared_theta = np.array([0.1, 0.2], dtype=np.float64)
    compound_theta = np.array([0.1, 0.2, 1.1, 1.2, 2.1, 2.2], dtype=np.float64)

    assert gpc_log_marginal_likelihood_use_shared_theta(shared_theta, 2, 3) is True
    assert gpc_log_marginal_likelihood_use_shared_theta(compound_theta, 2, 3) is False

    assert gpc_log_marginal_likelihood_use_compound_theta(shared_theta, 2, 3) is False
    assert gpc_log_marginal_likelihood_use_compound_theta(compound_theta, 2, 3) is True

    assert np.array_equal(gpc_log_marginal_likelihood_theta_slice(compound_theta, 2, 0), np.array([0.1, 0.2]))
    assert np.array_equal(gpc_log_marginal_likelihood_theta_slice(compound_theta, 2, 2), np.array([2.1, 2.2]))
    with pytest.raises(ValueError, match="theta does not contain a full slice for the requested estimator_index"):
        gpc_log_marginal_likelihood_theta_slice(shared_theta, 2, 1)

    assert gpc_log_marginal_likelihood_mean(np.array([1.0, 2.0, 4.0], dtype=np.float64)) == pytest.approx(7.0 / 3.0)
    assert (
        gpc_log_marginal_likelihood_theta_shape_message(2, 3, 5)
        == "Shape of theta must be either 2 or 6. Obtained theta with shape 5."
    )
