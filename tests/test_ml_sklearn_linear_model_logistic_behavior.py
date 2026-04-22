from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn._loss.loss import HalfBinomialLoss
from sklearn.linear_model._linear_loss import LinearModelLoss


def test_logistic_helper_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic import (
        binary_logistic_dense_loss_gradient,
        binary_logistic_half_loss_gradient,
        binary_logistic_positive_probability,
    )

    assert callable(binary_logistic_positive_probability)
    assert callable(binary_logistic_half_loss_gradient)
    assert callable(binary_logistic_dense_loss_gradient)


def test_binary_logistic_positive_probability_matches_sklearn_predict_proba() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic import binary_logistic_positive_probability

    raw_prediction = np.array([-40.0, -3.0, 0.0, 2.0, 20.0], dtype=np.float64)
    expected = HalfBinomialLoss().predict_proba(raw_prediction)[:, 1]

    assert np.allclose(binary_logistic_positive_probability(raw_prediction), expected)


def test_binary_logistic_half_loss_gradient_matches_sklearn_pointwise() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic import binary_logistic_half_loss_gradient

    y = np.array([0.0, 1.0, 0.25, 0.75, 1.0], dtype=np.float64)
    raw_prediction = np.array([-40.0, -3.0, 0.0, 2.0, 20.0], dtype=np.float64)
    sample_weight = np.array([1.0, 0.5, 2.0, 1.5, 0.25], dtype=np.float64)

    expected_loss, expected_gradient = HalfBinomialLoss().loss_gradient(
        y_true=y,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
        n_threads=1,
    )
    result_loss, result_gradient = binary_logistic_half_loss_gradient(
        y,
        raw_prediction,
        sample_weight=sample_weight,
    )

    assert np.allclose(result_loss, expected_loss)
    assert np.allclose(result_gradient, expected_gradient)


def test_binary_logistic_dense_loss_gradient_matches_sklearn_with_intercept() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic import binary_logistic_dense_loss_gradient

    X = np.array(
        [
            [1.0, 2.0, -0.5],
            [0.25, -1.0, 1.5],
            [2.0, 0.5, 0.0],
            [-1.0, 1.0, 2.5],
        ],
        dtype=np.float64,
    )
    y = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)
    params = np.array([0.4, -0.2, 0.1, 0.3], dtype=np.float64)
    sample_weight = np.array([1.0, 0.5, 2.0, 1.5], dtype=np.float64)

    expected_loss, expected_gradient = LinearModelLoss(
        base_loss=HalfBinomialLoss(),
        fit_intercept=True,
    ).loss_gradient(
        params,
        X,
        y,
        sample_weight=sample_weight,
        l2_reg_strength=0.05,
        n_threads=1,
    )
    result_loss, result_gradient = binary_logistic_dense_loss_gradient(
        params,
        X,
        y,
        alpha=0.05,
        sample_weight=sample_weight,
    )

    assert result_loss == pytest.approx(expected_loss)
    assert np.allclose(result_gradient, expected_gradient)


def test_binary_logistic_dense_loss_gradient_matches_sklearn_without_intercept() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic import binary_logistic_dense_loss_gradient

    X = np.array([[1.0, 2.0], [0.25, -1.0], [2.0, 0.5]], dtype=np.float64)
    y = np.array([1.0, 0.0, 1.0], dtype=np.float64)
    params = np.array([0.4, -0.2], dtype=np.float64)

    expected_loss, expected_gradient = LinearModelLoss(
        base_loss=HalfBinomialLoss(),
        fit_intercept=False,
    ).loss_gradient(
        params,
        X,
        y,
        sample_weight=None,
        l2_reg_strength=0.2,
        n_threads=1,
    )
    result_loss, result_gradient = binary_logistic_dense_loss_gradient(
        params,
        X,
        y,
        alpha=0.2,
    )

    assert result_loss == pytest.approx(expected_loss)
    assert np.allclose(result_gradient, expected_gradient)


def test_contracts_reject_invalid_logistic_inputs() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic import (
        binary_logistic_dense_loss_gradient,
        binary_logistic_half_loss_gradient,
        binary_logistic_positive_probability,
    )

    with pytest.raises(ViolationError):
        binary_logistic_positive_probability(np.array([0.0, np.inf], dtype=np.float64))

    with pytest.raises(ViolationError):
        binary_logistic_half_loss_gradient(
            np.array([0.0, 2.0], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
        )

    with pytest.raises(ViolationError):
        binary_logistic_half_loss_gradient(
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
            sample_weight=np.array([1.0, -1.0], dtype=np.float64),
        )

    with pytest.raises(ViolationError):
        binary_logistic_dense_loss_gradient(
            np.array([0.1], dtype=np.float64),
            np.ones((3, 2), dtype=np.float64),
            np.ones(3, dtype=np.float64),
        )
