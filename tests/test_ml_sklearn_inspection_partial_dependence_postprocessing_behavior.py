from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.inspection._partial_dependence import _partial_dependence_brute
from sklearn.linear_model import LinearRegression, LogisticRegression


def test_partial_dependence_postprocessing_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_postprocessing import (
        partial_dependence_assign_grid_values,
        partial_dependence_average_response_sequence,
        partial_dependence_finalize_averages,
        partial_dependence_finalize_predictions,
        partial_dependence_stack_response_sequence,
    )

    assert callable(partial_dependence_assign_grid_values)
    assert callable(partial_dependence_average_response_sequence)
    assert callable(partial_dependence_stack_response_sequence)
    assert callable(partial_dependence_finalize_predictions)
    assert callable(partial_dependence_finalize_averages)


def test_partial_dependence_assign_grid_values_updates_selected_columns_without_mutation() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_postprocessing import partial_dependence_assign_grid_values

    X = np.array(
        [
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
        ],
        dtype=np.float64,
    )

    updated = partial_dependence_assign_grid_values(
        X,
        np.array([10.0, 20.0], dtype=np.float64),
        features=(0, 2),
    )

    assert np.array_equal(X, np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float64))
    assert np.array_equal(updated[:, 0], np.array([10.0, 10.0], dtype=np.float64))
    assert np.array_equal(updated[:, 1], X[:, 1])
    assert np.array_equal(updated[:, 2], np.array([20.0, 20.0], dtype=np.float64))


def test_partial_dependence_postprocessing_regression_matches_private_helper() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_postprocessing import (
        partial_dependence_assign_grid_values,
        partial_dependence_average_response_sequence,
        partial_dependence_finalize_averages,
        partial_dependence_finalize_predictions,
        partial_dependence_stack_response_sequence,
    )

    X, y = load_diabetes(return_X_y=True)
    X = np.asarray(X[:24], dtype=np.float64)
    y = np.asarray(y[:24], dtype=np.float64)
    estimator = LinearRegression().fit(X, y)
    grid = np.array([[-0.05], [0.02], [0.07]], dtype=np.float64)

    expected_averages, expected_predictions = _partial_dependence_brute(
        estimator,
        grid,
        np.array([0]),
        X,
        response_method="auto",
    )

    responses = tuple(
        estimator.predict(
            partial_dependence_assign_grid_values(
                X,
                new_values,
                features=(0,),
            )
        )
        for new_values in grid
    )
    actual_average_stack = partial_dependence_average_response_sequence(responses)
    actual_prediction_stack = partial_dependence_stack_response_sequence(responses)

    actual_averages = partial_dependence_finalize_averages(actual_average_stack, task_kind="regression")
    actual_predictions = partial_dependence_finalize_predictions(
        actual_prediction_stack,
        task_kind="regression",
        n_samples=X.shape[0],
    )

    assert np.allclose(actual_averages, expected_averages)
    assert np.allclose(actual_predictions, expected_predictions)


def test_partial_dependence_postprocessing_binary_classifier_matches_private_helper() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_postprocessing import (
        partial_dependence_assign_grid_values,
        partial_dependence_average_response_sequence,
        partial_dependence_finalize_averages,
        partial_dependence_finalize_predictions,
        partial_dependence_stack_response_sequence,
    )

    X, y = load_breast_cancer(return_X_y=True)
    X = np.asarray(X[:32], dtype=np.float64)
    y = np.asarray(y[:32], dtype=np.int64)
    estimator = LogisticRegression(max_iter=1000).fit(X, y)
    sample_weight = np.linspace(1.0, 2.0, X.shape[0], dtype=np.float64)
    grid = np.array([[11.0], [15.0], [19.0]], dtype=np.float64)

    expected_averages, expected_predictions = _partial_dependence_brute(
        estimator,
        grid,
        np.array([0]),
        X,
        response_method="predict_proba",
        sample_weight=sample_weight,
    )

    responses = tuple(
        estimator.predict_proba(
            partial_dependence_assign_grid_values(
                X,
                new_values,
                features=(0,),
            )
        )
        for new_values in grid
    )
    actual_average_stack = partial_dependence_average_response_sequence(
        responses,
        sample_weight=sample_weight,
    )
    actual_prediction_stack = partial_dependence_stack_response_sequence(responses)

    actual_averages = partial_dependence_finalize_averages(actual_average_stack, task_kind="classification")
    actual_predictions = partial_dependence_finalize_predictions(
        actual_prediction_stack,
        task_kind="classification",
        n_samples=X.shape[0],
    )

    assert np.allclose(actual_averages, expected_averages)
    assert np.allclose(actual_predictions, expected_predictions)


def test_partial_dependence_postprocessing_multiclass_keeps_class_axis() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_postprocessing import (
        partial_dependence_assign_grid_values,
        partial_dependence_average_response_sequence,
        partial_dependence_finalize_averages,
        partial_dependence_finalize_predictions,
        partial_dependence_stack_response_sequence,
    )

    X, y = load_iris(return_X_y=True)
    X = np.asarray(X, dtype=np.float64)
    estimator = LogisticRegression(max_iter=1000).fit(X, y)
    grid = np.array([[4.5], [5.5]], dtype=np.float64)
    responses = tuple(
        estimator.predict_proba(
            partial_dependence_assign_grid_values(
                X,
                new_values,
                features=(0,),
            )
        )
        for new_values in grid
    )

    stacked_predictions = partial_dependence_stack_response_sequence(responses)
    stacked_averages = partial_dependence_average_response_sequence(responses)

    finalized_predictions = partial_dependence_finalize_predictions(
        stacked_predictions,
        task_kind="classification",
        n_samples=X.shape[0],
    )
    finalized_averages = partial_dependence_finalize_averages(
        stacked_averages,
        task_kind="classification",
    )

    assert finalized_predictions.shape == stacked_predictions.shape
    assert finalized_averages.shape == stacked_averages.shape
    assert finalized_predictions.shape[0] == 3
    assert finalized_averages.shape[0] == 3


def test_partial_dependence_postprocessing_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_postprocessing import (
        partial_dependence_assign_grid_values,
        partial_dependence_average_response_sequence,
        partial_dependence_finalize_predictions,
    )

    with pytest.raises(Exception):
        partial_dependence_assign_grid_values(
            np.ones((3, 2), dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            features=(0, 0),
        )

    with pytest.raises(Exception):
        partial_dependence_average_response_sequence(
            (np.ones(3, dtype=np.float64), np.ones(4, dtype=np.float64)),
        )

    with pytest.raises(Exception):
        partial_dependence_finalize_predictions(
            np.ones((2, 3), dtype=np.float64),
            task_kind="classification",
            n_samples=4,
        )
