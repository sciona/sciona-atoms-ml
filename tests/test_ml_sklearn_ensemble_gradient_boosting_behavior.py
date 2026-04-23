from __future__ import annotations

import math

import numpy as np
import pytest
from icontract import ViolationError
from sklearn._loss.loss import HuberLoss
from sklearn.ensemble._gb import _safe_divide, set_huber_delta


def test_gradient_boosting_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.ensemble.gradient_boosting import (
        gradient_boosting_huber_delta,
        gradient_boosting_safe_divide,
    )

    assert callable(gradient_boosting_safe_divide)
    assert callable(gradient_boosting_huber_delta)


@pytest.mark.parametrize(
    ("numerator", "denominator"),
    [
        (3.0, 2.0),
        (-5.5, 4.0),
        (7.0, 1e-200),
    ],
)
def test_gradient_boosting_safe_divide_matches_sklearn(numerator: float, denominator: float) -> None:
    from sciona.atoms.ml.sklearn.ensemble.gradient_boosting import gradient_boosting_safe_divide

    result = gradient_boosting_safe_divide(numerator, denominator)
    expected = _safe_divide(numerator, denominator)
    assert result == expected


def test_gradient_boosting_huber_delta_matches_sklearn_setter() -> None:
    from sciona.atoms.ml.sklearn.ensemble.gradient_boosting import gradient_boosting_huber_delta

    y_true = np.array([1.0, 2.5, 0.5, -1.0, 3.0], dtype=np.float64)
    raw_prediction = np.array([[0.8], [2.0], [1.0], [-0.2], [2.1]], dtype=np.float64)
    sample_weight = np.array([1.0, 2.0, 1.5, 0.5, 3.0], dtype=np.float64)
    quantile = 0.8

    loss = HuberLoss(quantile=quantile)
    set_huber_delta(loss, y_true, raw_prediction, sample_weight)

    result = gradient_boosting_huber_delta(
        y_true,
        raw_prediction,
        sample_weight,
        quantile=quantile,
    )
    assert math.isclose(result, float(loss.closs.delta))


def test_contracts_reject_invalid_gradient_boosting_inputs() -> None:
    from sciona.atoms.ml.sklearn.ensemble.gradient_boosting import (
        gradient_boosting_huber_delta,
        gradient_boosting_safe_divide,
    )

    with pytest.raises(ViolationError):
        gradient_boosting_safe_divide(float("nan"), 1.0)

    with pytest.raises(ViolationError):
        gradient_boosting_huber_delta(
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            np.array([1.0, 1.0], dtype=np.float64),
        )

    with pytest.raises(ViolationError):
        gradient_boosting_huber_delta(
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([0.5, 0.4], dtype=np.float64),
            np.array([1.0, -1.0], dtype=np.float64),
        )

    with pytest.raises(ViolationError):
        gradient_boosting_huber_delta(
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([0.5, 0.4], dtype=np.float64),
            np.array([1.0, 1.0], dtype=np.float64),
            quantile=1.0,
        )
