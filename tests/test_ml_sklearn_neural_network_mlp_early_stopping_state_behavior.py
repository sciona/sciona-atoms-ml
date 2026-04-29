from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError

from sciona.atoms.ml.sklearn.neural_network.mlp_early_stopping_state import (
    mlp_monitor_best_state,
    mlp_restore_best_parameters,
    mlp_stochastic_validation_targets,
    mlp_validation_scores_append,
)
from sciona.atoms.ml.sklearn.preprocessing.state_models import LabelBinarizerState


def _binary_label_binarizer_state() -> LabelBinarizerState:
    return LabelBinarizerState(
        classes=np.asarray(["neg", "pos"], dtype=object),
        y_type="binary",
        sparse_input=False,
        neg_label=0,
        pos_label=1,
        sparse_output=False,
    )


def _weights() -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    coefs = (
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        np.array([[5.0], [6.0]], dtype=np.float64),
    )
    intercepts = (
        np.array([0.1, 0.2], dtype=np.float64),
        np.array([0.3], dtype=np.float64),
    )
    return coefs, intercepts


def test_mlp_stochastic_validation_targets_decodes_classifier_targets() -> None:
    y_val = np.array([[0], [1], [1]], dtype=np.float64)
    observed = mlp_stochastic_validation_targets(
        y_val,
        is_classifier=True,
        label_binarizer_state=_binary_label_binarizer_state(),
    )
    assert np.array_equal(observed, np.array(["neg", "pos", "pos"], dtype=object))


def test_mlp_stochastic_validation_targets_passthrough_for_regression() -> None:
    y_val = np.array([[1.5], [2.5]], dtype=np.float64)
    observed = mlp_stochastic_validation_targets(y_val, is_classifier=False)
    assert np.array_equal(observed, y_val)


def test_mlp_validation_scores_append_appends_once() -> None:
    observed = mlp_validation_scores_append((0.3, 0.4), 0.5)
    assert observed == pytest.approx((0.3, 0.4, 0.5))


def test_mlp_monitor_best_state_updates_score_and_copies_params_on_improvement() -> None:
    best_coefs, best_intercepts = _weights()
    coefs, intercepts = _weights()
    observed_score, observed_coefs, observed_intercepts = mlp_monitor_best_state(
        0.9,
        0.7,
        best_coefs,
        best_intercepts,
        coefs,
        intercepts,
    )
    assert observed_score == pytest.approx(0.9)
    assert np.array_equal(observed_coefs[0], coefs[0])
    assert np.array_equal(observed_intercepts[1], intercepts[1])

    coefs[0][0, 0] = -100.0
    intercepts[0][0] = -200.0
    assert observed_coefs[0][0, 0] == pytest.approx(1.0)
    assert observed_intercepts[0][0] == pytest.approx(0.1)


def test_mlp_monitor_best_state_preserves_cached_params_without_improvement() -> None:
    best_coefs, best_intercepts = _weights()
    coefs, intercepts = _weights()
    coefs = tuple(array + 10.0 for array in coefs)
    intercepts = tuple(array + 10.0 for array in intercepts)
    observed_score, observed_coefs, observed_intercepts = mlp_monitor_best_state(
        0.6,
        0.7,
        best_coefs,
        best_intercepts,
        coefs,
        intercepts,
    )
    assert observed_score == pytest.approx(0.7)
    assert np.array_equal(observed_coefs[0], best_coefs[0])
    assert np.array_equal(observed_intercepts[1], best_intercepts[1])


def test_mlp_restore_best_parameters_returns_cached_values() -> None:
    best_coefs, best_intercepts = _weights()
    observed_coefs, observed_intercepts = mlp_restore_best_parameters(
        best_coefs,
        best_intercepts,
    )
    assert np.array_equal(observed_coefs[1], best_coefs[1])
    assert np.array_equal(observed_intercepts[0], best_intercepts[0])


def test_mlp_early_stopping_state_rejects_invalid_inputs() -> None:
    with pytest.raises((ViolationError, ValueError)):
        mlp_stochastic_validation_targets(
            np.array([[0.0], [1.0]], dtype=np.float64),
            is_classifier=True,
        )

    with pytest.raises((ViolationError, ValueError)):
        mlp_validation_scores_append((0.1,), float("nan"))

    coefs, intercepts = _weights()
    with pytest.raises((ViolationError, ValueError)):
        mlp_monitor_best_state(
            0.5,
            float("-inf"),
            coefs,
            intercepts[:-1],
            coefs,
            intercepts,
        )
