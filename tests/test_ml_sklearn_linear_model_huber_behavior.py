from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.linear_model._huber import _huber_loss_and_gradient


def test_huber_helper_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.huber import (
        huber_linear_residuals,
        huber_loss_gradient,
        huber_outlier_mask,
    )

    assert callable(huber_linear_residuals)
    assert callable(huber_outlier_mask)
    assert callable(huber_loss_gradient)


def test_huber_linear_residuals_and_outlier_mask_match_source_steps() -> None:
    from sciona.atoms.ml.sklearn.linear_model.huber import huber_linear_residuals, huber_outlier_mask

    X = np.array([[1.0, 2.0], [0.5, -1.0], [3.0, 0.25]], dtype=np.float64)
    y = np.array([1.5, -0.25, 3.0], dtype=np.float64)
    coef = np.array([0.75, -0.5], dtype=np.float64)
    intercept = 0.2
    sigma = 0.9
    epsilon = 1.35

    residuals = huber_linear_residuals(X, y, coef, intercept=intercept)
    expected_residuals = y - X.dot(coef) - intercept
    assert np.allclose(residuals, expected_residuals)
    assert np.array_equal(huber_outlier_mask(residuals, epsilon=epsilon, sigma=sigma), np.abs(expected_residuals) > epsilon * sigma)


def test_huber_loss_gradient_matches_sklearn_private_helper_with_intercept() -> None:
    from sciona.atoms.ml.sklearn.linear_model.huber import huber_loss_gradient

    X = np.array(
        [
            [1.0, 2.0, -0.5],
            [0.25, -1.0, 1.5],
            [2.0, 0.5, 0.0],
            [-1.0, 1.0, 2.5],
        ],
        dtype=np.float64,
    )
    y = np.array([1.0, -2.0, 3.5, 0.25], dtype=np.float64)
    params = np.array([0.4, -0.8, 0.2, 0.15, 1.1], dtype=np.float64)
    sample_weight = np.array([1.0, 0.5, 2.0, 1.5], dtype=np.float64)

    expected_loss, expected_gradient = _huber_loss_and_gradient(
        params,
        X,
        y,
        epsilon=1.35,
        alpha=0.01,
        sample_weight=sample_weight,
    )
    result_loss, result_gradient = huber_loss_gradient(
        params,
        X,
        y,
        epsilon=1.35,
        alpha=0.01,
        sample_weight=sample_weight,
    )

    assert result_loss == pytest.approx(expected_loss)
    assert np.allclose(result_gradient, expected_gradient)


def test_huber_loss_gradient_matches_sklearn_private_helper_without_intercept() -> None:
    from sciona.atoms.ml.sklearn.linear_model.huber import huber_loss_gradient

    X = np.array([[1.0, 2.0], [0.25, -1.0], [2.0, 0.5]], dtype=np.float64)
    y = np.array([1.0, -2.0, 3.5], dtype=np.float64)
    params = np.array([0.4, -0.8, 1.1], dtype=np.float64)
    sample_weight = np.ones(X.shape[0], dtype=np.float64)

    expected_loss, expected_gradient = _huber_loss_and_gradient(
        params,
        X,
        y,
        epsilon=1.5,
        alpha=0.2,
        sample_weight=sample_weight,
    )
    result_loss, result_gradient = huber_loss_gradient(params, X, y, epsilon=1.5, alpha=0.2)

    assert result_loss == pytest.approx(expected_loss)
    assert np.allclose(result_gradient, expected_gradient)


def test_contracts_reject_invalid_huber_inputs() -> None:
    from sciona.atoms.ml.sklearn.linear_model.huber import (
        huber_linear_residuals,
        huber_loss_gradient,
        huber_outlier_mask,
    )

    with pytest.raises(ViolationError):
        huber_linear_residuals(np.ones((2, 2), dtype=np.float64), np.ones(3, dtype=np.float64), np.ones(2, dtype=np.float64))

    with pytest.raises(ViolationError):
        huber_outlier_mask(np.ones(3, dtype=np.float64), epsilon=0.5, sigma=1.0)

    with pytest.raises(ViolationError):
        huber_loss_gradient(
            np.array([1.0, 2.0], dtype=np.float64),
            np.ones((3, 2), dtype=np.float64),
            np.ones(3, dtype=np.float64),
            epsilon=1.35,
            alpha=0.0,
        )

    with pytest.raises(ViolationError):
        huber_loss_gradient(
            np.array([1.0, 2.0, -1.0], dtype=np.float64),
            np.ones((3, 2), dtype=np.float64),
            np.ones(3, dtype=np.float64),
            epsilon=1.35,
            alpha=0.0,
        )
