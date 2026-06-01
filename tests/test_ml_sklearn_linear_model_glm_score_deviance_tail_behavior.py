from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.linear_model import PoissonRegressor


def test_glm_score_deviance_tail_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.glm_score_deviance_tail import (
        glm_score_constant_average,
        glm_score_d2_from_deviances,
        glm_score_null_raw_prediction,
        glm_score_sample_weight_check_args,
        glm_score_sample_weight_check_kwargs,
        glm_score_target_range_error_message,
        glm_score_y_check_array_kwargs,
    )

    assert callable(glm_score_y_check_array_kwargs)
    assert callable(glm_score_sample_weight_check_args)
    assert callable(glm_score_sample_weight_check_kwargs)
    assert callable(glm_score_target_range_error_message)
    assert callable(glm_score_constant_average)
    assert callable(glm_score_null_raw_prediction)
    assert callable(glm_score_d2_from_deviances)


def test_glm_score_validation_payloads_match_source_kwargs() -> None:
    from sciona.atoms.ml.sklearn.linear_model.glm_score_deviance_tail import (
        glm_score_sample_weight_check_args,
        glm_score_sample_weight_check_kwargs,
        glm_score_y_check_array_kwargs,
    )

    raw_prediction = np.array([0.1, 0.2], dtype=np.float32)
    y = np.array([1.0, 2.0], dtype=np.float32)
    sample_weight = np.array([1.0, 2.0], dtype=np.float64)
    X = object()

    assert glm_score_y_check_array_kwargs(raw_prediction) == {
        "dtype": raw_prediction.dtype,
        "order": "C",
        "ensure_2d": False,
    }
    args = glm_score_sample_weight_check_args(sample_weight, X)
    assert args == (sample_weight, X)
    assert args[0] is sample_weight
    assert args[1] is X
    assert glm_score_sample_weight_check_kwargs(y) == {"dtype": y.dtype}


def test_glm_score_message_average_null_prediction_and_d2_formula() -> None:
    from sciona.atoms.ml.sklearn.linear_model.glm_score_deviance_tail import (
        glm_score_constant_average,
        glm_score_d2_from_deviances,
        glm_score_null_raw_prediction,
        glm_score_target_range_error_message,
    )

    constants = np.array([0.0, 1.0, 3.0], dtype=np.float64)
    weights = np.array([1.0, 2.0, 1.0], dtype=np.float64)

    assert glm_score_target_range_error_message("HalfPoissonLoss") == (
        "Some value(s) of y are out of the valid range of the loss HalfPoissonLoss."
    )
    assert glm_score_constant_average(constants, weights) == pytest.approx(np.average(constants, weights=weights))
    np.testing.assert_allclose(glm_score_null_raw_prediction(np.array([1.0, 2.0, 3.0]), 0.75), np.array([0.75, 0.75, 0.75]))
    assert glm_score_d2_from_deviances(2.0, 8.0, 1.0) == pytest.approx(1.0 - 3.0 / 9.0)


def test_glm_score_deviance_tail_reconstructs_poisson_score_from_supplied_callbacks() -> None:
    from sciona.atoms.ml.sklearn.linear_model.glm_score_deviance_tail import (
        glm_score_constant_average,
        glm_score_d2_from_deviances,
        glm_score_null_raw_prediction,
    )

    X = np.array([[0.0, 1.0], [1.0, -0.5], [2.0, 0.25], [3.0, 1.5]], dtype=np.float64)
    y = np.array([1.0, 2.0, 4.0, 7.0], dtype=np.float64)
    sample_weight = np.array([1.0, 2.0, 1.0, 1.5], dtype=np.float64)
    estimator = PoissonRegressor(alpha=0.0, max_iter=300).fit(X, y, sample_weight=sample_weight)
    base_loss = estimator._base_loss
    raw_prediction = estimator._linear_predictor(X)

    constant = glm_score_constant_average(
        base_loss.constant_to_optimal_zero(y_true=y, sample_weight=None),
        sample_weight,
    )
    deviance = base_loss(y_true=y, raw_prediction=raw_prediction, sample_weight=sample_weight, n_threads=1)
    linked_mean = base_loss.link.link(np.average(y, weights=sample_weight))
    null_prediction = glm_score_null_raw_prediction(y, linked_mean)
    deviance_null = base_loss(y_true=y, raw_prediction=null_prediction, sample_weight=sample_weight, n_threads=1)

    assert glm_score_d2_from_deviances(deviance, deviance_null, constant) == pytest.approx(
        estimator.score(X, y, sample_weight=sample_weight)
    )


def test_glm_score_deviance_tail_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.glm_score_deviance_tail import (
        glm_score_constant_average,
        glm_score_d2_from_deviances,
        glm_score_null_raw_prediction,
        glm_score_sample_weight_check_args,
        glm_score_sample_weight_check_kwargs,
        glm_score_target_range_error_message,
        glm_score_y_check_array_kwargs,
    )

    with pytest.raises(ViolationError):
        glm_score_y_check_array_kwargs(object())

    with pytest.raises(ViolationError):
        glm_score_sample_weight_check_args(None, object())

    with pytest.raises(ViolationError):
        glm_score_sample_weight_check_kwargs(SimpleNamespace())

    with pytest.raises(ViolationError):
        glm_score_target_range_error_message("")

    with pytest.raises(ViolationError):
        glm_score_constant_average(np.array([1.0, 2.0]), np.array([0.0, 0.0]))

    with pytest.raises(ViolationError):
        glm_score_null_raw_prediction(np.array([1.0, np.nan]), 0.0)

    with pytest.raises(ViolationError):
        glm_score_d2_from_deviances(1.0, -1.0, 1.0)
