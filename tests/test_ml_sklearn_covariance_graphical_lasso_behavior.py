from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.covariance._empirical_covariance import log_likelihood
from sklearn.covariance._graph_lasso import _dual_gap, _objective


def _empirical_covariance() -> np.ndarray:
    return np.array(
        [
            [2.0, 0.3, -0.2],
            [0.3, 1.5, 0.4],
            [-0.2, 0.4, 1.2],
        ],
        dtype=np.float64,
    )


def _precision() -> np.ndarray:
    covariance = np.array(
        [
            [1.6, 0.25, 0.1],
            [0.25, 1.4, -0.15],
            [0.1, -0.15, 1.1],
        ],
        dtype=np.float64,
    )
    return np.asarray(np.linalg.inv(covariance), dtype=np.float64)


def test_graphical_lasso_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.covariance.graphical_lasso import (
        graphical_lasso_dual_gap,
        graphical_lasso_log_likelihood,
        graphical_lasso_objective,
        graphical_lasso_offdiag_l1_penalty,
    )

    assert callable(graphical_lasso_offdiag_l1_penalty)
    assert callable(graphical_lasso_log_likelihood)
    assert callable(graphical_lasso_objective)
    assert callable(graphical_lasso_dual_gap)


def test_offdiag_l1_penalty_excludes_diagonal() -> None:
    from sciona.atoms.ml.sklearn.covariance.graphical_lasso import graphical_lasso_offdiag_l1_penalty

    precision = _precision()
    expected = np.abs(precision).sum() - np.abs(np.diag(precision)).sum()

    assert graphical_lasso_offdiag_l1_penalty(precision) == pytest.approx(expected)


def test_log_likelihood_matches_sklearn_covariance_helper() -> None:
    from sciona.atoms.ml.sklearn.covariance.graphical_lasso import graphical_lasso_log_likelihood

    emp_cov = _empirical_covariance()
    precision = _precision()

    assert graphical_lasso_log_likelihood(emp_cov, precision) == pytest.approx(log_likelihood(emp_cov, precision))


def test_objective_and_dual_gap_match_sklearn_private_helpers() -> None:
    from sciona.atoms.ml.sklearn.covariance.graphical_lasso import (
        graphical_lasso_dual_gap,
        graphical_lasso_objective,
    )

    emp_cov = _empirical_covariance()
    precision = _precision()
    alpha = 0.12

    assert graphical_lasso_objective(emp_cov, precision, alpha) == pytest.approx(_objective(emp_cov, precision, alpha))
    assert graphical_lasso_dual_gap(emp_cov, precision, alpha) == pytest.approx(_dual_gap(emp_cov, precision, alpha))


def test_objective_and_dual_gap_handle_unpenalized_alpha() -> None:
    from sciona.atoms.ml.sklearn.covariance.graphical_lasso import (
        graphical_lasso_dual_gap,
        graphical_lasso_objective,
    )

    emp_cov = _empirical_covariance()
    precision = _precision()

    assert graphical_lasso_objective(emp_cov, precision, 0.0) == pytest.approx(_objective(emp_cov, precision, 0.0))
    assert graphical_lasso_dual_gap(emp_cov, precision, 0.0) == pytest.approx(_dual_gap(emp_cov, precision, 0.0))


def test_contracts_reject_invalid_graphical_lasso_inputs() -> None:
    from sciona.atoms.ml.sklearn.covariance.graphical_lasso import (
        graphical_lasso_dual_gap,
        graphical_lasso_log_likelihood,
        graphical_lasso_objective,
        graphical_lasso_offdiag_l1_penalty,
    )

    emp_cov = _empirical_covariance()
    precision = _precision()

    with pytest.raises(ViolationError):
        graphical_lasso_offdiag_l1_penalty(np.array([[1.0, np.nan]], dtype=np.float64))

    with pytest.raises(ViolationError):
        graphical_lasso_log_likelihood(emp_cov, np.diag([1.0, 0.0, 2.0]))

    with pytest.raises(ViolationError):
        graphical_lasso_objective(emp_cov, precision, -0.1)

    with pytest.raises(ViolationError):
        graphical_lasso_dual_gap(emp_cov[:2, :2], precision, 0.1)
