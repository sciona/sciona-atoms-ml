from __future__ import annotations

import math

import numpy as np
import pytest
from icontract import ViolationError

from sciona.atoms.ml.sklearn.neural_network.mlp_regression_io import (
    mlp_regressor_predictions,
    mlp_regressor_r2_score,
    mlp_regressor_targets,
)


def test_mlp_regressor_targets_flattens_single_target_column() -> None:
    y = np.array([[1.0], [2.5], [3.0]], dtype=np.float64)
    observed = mlp_regressor_targets(y)
    assert observed.shape == (3,)
    assert np.array_equal(observed, np.array([1.0, 2.5, 3.0], dtype=np.float64))


def test_mlp_regressor_targets_preserves_multioutput_targets() -> None:
    y = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    observed = mlp_regressor_targets(y)
    assert observed.shape == (2, 2)
    assert np.array_equal(observed, y)


def test_mlp_regressor_predictions_flattens_one_output_matrix() -> None:
    y_pred = np.array([[0.5], [1.5]], dtype=np.float64)
    observed = mlp_regressor_predictions(y_pred)
    assert observed.shape == (2,)
    assert np.array_equal(observed, np.array([0.5, 1.5], dtype=np.float64))


def test_mlp_regressor_predictions_preserves_multioutput_matrix() -> None:
    y_pred = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float64)
    observed = mlp_regressor_predictions(y_pred)
    assert observed.shape == (2, 2)
    assert np.array_equal(observed, y_pred)


def test_mlp_regressor_r2_score_matches_sklearn_for_finite_predictions() -> None:
    y_true = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y_pred = np.array([1.0, 2.5, 2.5], dtype=np.float64)
    observed = mlp_regressor_r2_score(y_true, y_pred)
    assert math.isclose(observed, 0.75)


def test_mlp_regressor_r2_score_short_circuits_on_invalid_predictions() -> None:
    y_true = np.array([1.0, 2.0], dtype=np.float64)
    y_pred = np.array([1.0, np.nan], dtype=np.float64)
    observed = mlp_regressor_r2_score(y_true, y_pred)
    assert math.isnan(observed)


def test_mlp_regression_io_rejects_invalid_inputs() -> None:
    with pytest.raises((ViolationError, ValueError)):
        mlp_regressor_predictions(np.array([1.0, 2.0], dtype=np.float64))

    with pytest.raises((ViolationError, ValueError)):
        mlp_regressor_r2_score(
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([[1.0], [2.0]], dtype=np.float64),
        )
