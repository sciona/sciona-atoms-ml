from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_quantile_helper_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.quantile import (
        quantile_dense_lp_problem,
        quantile_nonzero_weight_mask,
        quantile_solution_to_params,
    )

    assert callable(quantile_nonzero_weight_mask)
    assert callable(quantile_dense_lp_problem)
    assert callable(quantile_solution_to_params)


def test_quantile_nonzero_weight_mask_matches_source_filter() -> None:
    from sciona.atoms.ml.sklearn.linear_model.quantile import quantile_nonzero_weight_mask

    sample_weight = np.array([1.0, 0.0, 2.5, 0.0, 3.0], dtype=np.float64)

    assert np.array_equal(quantile_nonzero_weight_mask(sample_weight), sample_weight != 0.0)


def test_quantile_dense_lp_problem_matches_sklearn_dense_form_with_intercept() -> None:
    from sciona.atoms.ml.sklearn.linear_model.quantile import quantile_dense_lp_problem

    X = np.array([[1.0, 2.0], [0.5, -1.0], [3.0, 0.25], [-2.0, 1.0]], dtype=np.float64)
    y = np.array([1.5, -0.25, 3.0, 0.75], dtype=np.float64)
    sample_weight = np.array([1.0, 0.0, 2.0, 1.5], dtype=np.float64)
    quantile = 0.8
    alpha = 0.2

    c, a_eq, b_eq = quantile_dense_lp_problem(
        X,
        y,
        sample_weight,
        quantile=quantile,
        alpha=alpha,
        fit_intercept=True,
    )

    mask = sample_weight != 0.0
    X_used = X[mask]
    y_used = y[mask]
    weights_used = sample_weight[mask]
    n_rows, n_features = X_used.shape
    n_params = n_features + 1
    expected_c = np.concatenate(
        [
            np.full(2 * n_params, fill_value=np.sum(weights_used) * alpha),
            weights_used * quantile,
            weights_used * (1.0 - quantile),
        ]
    )
    expected_c[0] = 0.0
    expected_c[n_params] = 0.0
    eye = np.eye(n_rows)
    ones = np.ones((n_rows, 1))
    expected_a_eq = np.concatenate([ones, X_used, -ones, -X_used, eye, -eye], axis=1)

    assert np.allclose(c, expected_c)
    assert np.allclose(a_eq, expected_a_eq)
    assert np.allclose(b_eq, y_used)


def test_quantile_dense_lp_problem_matches_sklearn_dense_form_without_intercept() -> None:
    from sciona.atoms.ml.sklearn.linear_model.quantile import quantile_dense_lp_problem

    X = np.array([[1.0, 2.0], [0.5, -1.0], [3.0, 0.25]], dtype=np.float64)
    y = np.array([1.5, -0.25, 3.0], dtype=np.float64)
    sample_weight = np.array([1.0, 2.0, 1.5], dtype=np.float64)

    c, a_eq, b_eq = quantile_dense_lp_problem(
        X,
        y,
        sample_weight,
        quantile=0.25,
        alpha=0.1,
        fit_intercept=False,
    )

    n_rows, n_features = X.shape
    expected_c = np.concatenate(
        [
            np.full(2 * n_features, fill_value=np.sum(sample_weight) * 0.1),
            sample_weight * 0.25,
            sample_weight * 0.75,
        ]
    )
    eye = np.eye(n_rows)
    expected_a_eq = np.concatenate([X, -X, eye, -eye], axis=1)

    assert np.allclose(c, expected_c)
    assert np.allclose(a_eq, expected_a_eq)
    assert np.allclose(b_eq, y)


def test_quantile_solution_to_params_matches_source_decoding() -> None:
    from sciona.atoms.ml.sklearn.linear_model.quantile import quantile_solution_to_params

    solution = np.array([2.0, 1.5, 0.0, 0.25, 0.5, 3.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    coef, intercept = quantile_solution_to_params(solution, 2, fit_intercept=True)

    params = solution[:3] - solution[3:6]
    assert intercept == pytest.approx(params[0])
    assert np.allclose(coef, params[1:])

    coef_no_intercept, intercept_no_intercept = quantile_solution_to_params(solution, 2, fit_intercept=False)
    params_no_intercept = solution[:2] - solution[2:4]
    assert intercept_no_intercept == 0.0
    assert np.allclose(coef_no_intercept, params_no_intercept)


def test_contracts_reject_invalid_quantile_inputs() -> None:
    from sciona.atoms.ml.sklearn.linear_model.quantile import (
        quantile_dense_lp_problem,
        quantile_nonzero_weight_mask,
        quantile_solution_to_params,
    )

    with pytest.raises(ViolationError):
        quantile_nonzero_weight_mask(np.array([0.0, 0.0], dtype=np.float64))

    with pytest.raises(ViolationError):
        quantile_dense_lp_problem(
            np.ones((2, 2), dtype=np.float64),
            np.ones(3, dtype=np.float64),
            np.ones(2, dtype=np.float64),
            quantile=0.5,
            alpha=0.0,
        )

    with pytest.raises(ViolationError):
        quantile_dense_lp_problem(
            np.ones((2, 2), dtype=np.float64),
            np.ones(2, dtype=np.float64),
            np.ones(2, dtype=np.float64),
            quantile=1.0,
            alpha=0.0,
        )

    with pytest.raises(ViolationError):
        quantile_solution_to_params(np.ones(3, dtype=np.float64), 2)
