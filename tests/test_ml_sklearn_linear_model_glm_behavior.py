from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn._loss.loss import HalfGammaLoss, HalfPoissonLoss, HalfTweedieLoss
from sklearn.linear_model._linear_loss import LinearModelLoss


def test_glm_helper_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.glm import (
        glm_dense_loss_gradient,
        glm_linear_raw_prediction,
        glm_log_link_half_loss_gradient,
    )

    assert callable(glm_linear_raw_prediction)
    assert callable(glm_log_link_half_loss_gradient)
    assert callable(glm_dense_loss_gradient)


def test_glm_linear_raw_prediction_matches_source_formula() -> None:
    from sciona.atoms.ml.sklearn.linear_model.glm import glm_linear_raw_prediction

    X = np.array([[1.0, 2.0], [0.5, -1.0], [3.0, 0.25]], dtype=np.float64)
    coef = np.array([0.75, -0.5], dtype=np.float64)
    intercept = 0.2

    result = glm_linear_raw_prediction(X, coef, intercept=intercept)

    assert np.allclose(result, X @ coef + intercept)


@pytest.mark.parametrize(
    ("family", "base_loss", "power", "y"),
    [
        ("poisson", HalfPoissonLoss(), 1.5, np.array([0.0, 2.0, 5.0], dtype=np.float64)),
        ("gamma", HalfGammaLoss(), 1.5, np.array([1.0, 2.0, 5.0], dtype=np.float64)),
        ("tweedie", HalfTweedieLoss(power=1.5), 1.5, np.array([0.0, 2.0, 5.0], dtype=np.float64)),
    ],
)
def test_glm_log_link_half_loss_gradient_matches_sklearn_pointwise(
    family: str,
    base_loss: object,
    power: float,
    y: np.ndarray,
) -> None:
    from sciona.atoms.ml.sklearn.linear_model.glm import glm_log_link_half_loss_gradient

    raw_prediction = np.array([-0.7, 0.1, 1.2], dtype=np.float64)
    sample_weight = np.array([1.0, 0.5, 2.0], dtype=np.float64)

    expected_loss, expected_gradient = base_loss.loss_gradient(
        y_true=y,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
        n_threads=1,
    )
    result_loss, result_gradient = glm_log_link_half_loss_gradient(
        y,
        raw_prediction,
        family=family,
        power=power,
        sample_weight=sample_weight,
    )

    assert np.allclose(result_loss, expected_loss)
    assert np.allclose(result_gradient, expected_gradient)


@pytest.mark.parametrize(
    ("family", "base_loss", "power", "y"),
    [
        ("poisson", HalfPoissonLoss(), 1.5, np.array([1.0, 0.0, 3.0, 2.0], dtype=np.float64)),
        ("gamma", HalfGammaLoss(), 1.5, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)),
        ("tweedie", HalfTweedieLoss(power=1.5), 1.5, np.array([1.0, 0.0, 3.0, 2.0], dtype=np.float64)),
    ],
)
def test_glm_dense_loss_gradient_matches_sklearn_with_intercept(
    family: str,
    base_loss: object,
    power: float,
    y: np.ndarray,
) -> None:
    from sciona.atoms.ml.sklearn.linear_model.glm import glm_dense_loss_gradient

    X = np.array(
        [
            [1.0, 2.0, -0.5],
            [0.25, -1.0, 1.5],
            [2.0, 0.5, 0.0],
            [-1.0, 1.0, 2.5],
        ],
        dtype=np.float64,
    )
    params = np.array([0.4, -0.2, 0.1, 0.3], dtype=np.float64)
    sample_weight = np.array([1.0, 0.5, 2.0, 1.5], dtype=np.float64)

    expected_loss, expected_gradient = LinearModelLoss(
        base_loss=base_loss,
        fit_intercept=True,
    ).loss_gradient(
        params,
        X,
        y,
        sample_weight=sample_weight,
        l2_reg_strength=0.05,
        n_threads=1,
    )
    result_loss, result_gradient = glm_dense_loss_gradient(
        params,
        X,
        y,
        family=family,
        alpha=0.05,
        power=power,
        sample_weight=sample_weight,
    )

    assert result_loss == pytest.approx(expected_loss)
    assert np.allclose(result_gradient, expected_gradient)


def test_glm_dense_loss_gradient_matches_sklearn_without_intercept() -> None:
    from sciona.atoms.ml.sklearn.linear_model.glm import glm_dense_loss_gradient

    X = np.array([[1.0, 2.0], [0.25, -1.0], [2.0, 0.5]], dtype=np.float64)
    y = np.array([1.0, 0.0, 3.0], dtype=np.float64)
    params = np.array([0.4, -0.2], dtype=np.float64)

    expected_loss, expected_gradient = LinearModelLoss(
        base_loss=HalfPoissonLoss(),
        fit_intercept=False,
    ).loss_gradient(
        params,
        X,
        y,
        sample_weight=None,
        l2_reg_strength=0.2,
        n_threads=1,
    )
    result_loss, result_gradient = glm_dense_loss_gradient(
        params,
        X,
        y,
        family="poisson",
        alpha=0.2,
    )

    assert result_loss == pytest.approx(expected_loss)
    assert np.allclose(result_gradient, expected_gradient)


def test_contracts_reject_invalid_glm_inputs() -> None:
    from sciona.atoms.ml.sklearn.linear_model.glm import (
        glm_dense_loss_gradient,
        glm_linear_raw_prediction,
        glm_log_link_half_loss_gradient,
    )

    with pytest.raises(ViolationError):
        glm_linear_raw_prediction(np.ones((2, 2), dtype=np.float64), np.ones(3, dtype=np.float64))

    with pytest.raises(ViolationError):
        glm_log_link_half_loss_gradient(
            np.array([-1.0, 2.0], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
            family="poisson",
        )

    with pytest.raises(ViolationError):
        glm_log_link_half_loss_gradient(
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
            family="gamma",
            sample_weight=np.array([1.0, -1.0], dtype=np.float64),
        )

    with pytest.raises(ViolationError):
        glm_dense_loss_gradient(
            np.array([0.1], dtype=np.float64),
            np.ones((3, 2), dtype=np.float64),
            np.ones(3, dtype=np.float64),
            family="gamma",
        )
