from __future__ import annotations

import warnings
from types import MethodType

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier


def _binary_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    y = np.array([0, 1, 1, 0], dtype=np.int64)
    return X, y


def _multiclass_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ],
        dtype=np.float64,
    )
    y = np.array([0, 1, 2, 1, 2, 0], dtype=np.int64)
    return X, y


def _multilabel_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    y = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 1],
            [0, 0],
        ],
        dtype=np.int64,
    )
    return X, y


def _fit_binary_classifier() -> tuple[MLPClassifier, np.ndarray, np.ndarray]:
    X, y = _binary_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = MLPClassifier(
            hidden_layer_sizes=(3,),
            activation="tanh",
            solver="lbfgs",
            alpha=0.01,
            random_state=0,
            max_iter=60,
        ).fit(X, y)
    return model, X, y


def _fit_multiclass_classifier() -> tuple[MLPClassifier, np.ndarray, np.ndarray]:
    X, y = _multiclass_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = MLPClassifier(
            hidden_layer_sizes=(4,),
            activation="relu",
            solver="lbfgs",
            alpha=0.02,
            random_state=1,
            max_iter=80,
        ).fit(X, y)
    return model, X, y


def test_mlp_classification_io_import() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_classification_io import (
        mlp_classifier_encode_targets,
        mlp_classifier_fit_target_state,
        mlp_classifier_labels_from_outputs,
        mlp_classifier_partial_fit_target_state,
        mlp_classifier_probabilities_from_outputs,
    )

    assert callable(mlp_classifier_fit_target_state)
    assert callable(mlp_classifier_partial_fit_target_state)
    assert callable(mlp_classifier_encode_targets)
    assert callable(mlp_classifier_labels_from_outputs)
    assert callable(mlp_classifier_probabilities_from_outputs)


def test_mlp_classifier_fit_target_state_matches_private_validate_input_initial_fit() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_classification_io import (
        mlp_classifier_encode_targets,
        mlp_classifier_fit_target_state,
    )

    X, y = _binary_data()
    model = MLPClassifier(random_state=0)
    _, expected_y = model._validate_input(X, y, incremental=False, reset=True)

    state = mlp_classifier_fit_target_state(y)
    actual_y = mlp_classifier_encode_targets(y, state)

    assert np.array_equal(state.classes, model.classes_)
    assert state.y_type == model._label_binarizer.y_type_
    assert np.array_equal(actual_y, expected_y)


def test_mlp_classifier_fit_target_state_respects_warm_start_class_check() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_classification_io import mlp_classifier_fit_target_state

    _, y = _binary_data()
    existing_state = mlp_classifier_fit_target_state(y)

    with pytest.raises(ValueError, match="warm_start can only be used"):
        mlp_classifier_fit_target_state(np.array([0, 1, 2], dtype=np.int64), existing_state=existing_state, warm_start=True)


def test_mlp_classifier_fit_target_state_respects_incremental_class_check() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_classification_io import mlp_classifier_fit_target_state

    _, y = _binary_data()
    existing_state = mlp_classifier_fit_target_state(y)

    with pytest.raises(ValueError, match="`y` has classes not in `self.classes_`"):
        mlp_classifier_fit_target_state(np.array([0, 1, 2], dtype=np.int64), existing_state=existing_state, incremental=True)


def test_mlp_classifier_partial_fit_target_state_matches_partial_fit_first_call() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_classification_io import mlp_classifier_partial_fit_target_state

    X, y = _multiclass_data()
    classes = np.array([0, 1, 2], dtype=np.int64)
    model = MLPClassifier(random_state=0, solver="sgd")

    def fake_fit(self, X_input: np.ndarray, y_input: np.ndarray, incremental: bool) -> MLPClassifier:
        del X_input, y_input, incremental
        return self

    model._fit = MethodType(fake_fit, model)
    model.partial_fit(X, y, classes=classes)

    state = mlp_classifier_partial_fit_target_state(y, classes)

    assert np.array_equal(state.classes, model._label_binarizer.classes_)
    assert state.y_type == model._label_binarizer.y_type_


def test_mlp_classifier_partial_fit_target_state_multilabel_matches_partial_fit_branch() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_classification_io import mlp_classifier_partial_fit_target_state

    X, y = _multilabel_data()
    classes = np.array([0, 1], dtype=np.int64)
    model = MLPClassifier(random_state=0, solver="sgd")

    def fake_fit(self, X_input: np.ndarray, y_input: np.ndarray, incremental: bool) -> MLPClassifier:
        del X_input, y_input, incremental
        return self

    model._fit = MethodType(fake_fit, model)
    model.partial_fit(X, y, classes=classes)

    state = mlp_classifier_partial_fit_target_state(y, classes)

    assert np.array_equal(state.classes, model._label_binarizer.classes_)
    assert state.y_type == model._label_binarizer.y_type_


def test_mlp_classifier_labels_from_outputs_matches_private_predict_binary() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_classification_io import (
        mlp_classifier_fit_target_state,
        mlp_classifier_labels_from_outputs,
    )

    model, X, y = _fit_binary_classifier()
    outputs = model._forward_pass_fast(X)
    state = mlp_classifier_fit_target_state(y)

    actual = mlp_classifier_labels_from_outputs(outputs, state, n_outputs=model.n_outputs_)
    expected = model._predict(X, check_input=False)

    assert np.array_equal(actual, expected)


def test_mlp_classifier_labels_from_outputs_matches_private_predict_multiclass() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_classification_io import (
        mlp_classifier_fit_target_state,
        mlp_classifier_labels_from_outputs,
    )

    model, X, y = _fit_multiclass_classifier()
    outputs = model._forward_pass_fast(X)
    state = mlp_classifier_fit_target_state(y)

    actual = mlp_classifier_labels_from_outputs(outputs, state, n_outputs=model.n_outputs_)
    expected = model._predict(X, check_input=False)

    assert np.array_equal(actual, expected)


def test_mlp_classifier_probabilities_from_outputs_match_predict_proba_binary_and_multiclass() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_classification_io import mlp_classifier_probabilities_from_outputs

    binary_model, binary_X, _ = _fit_binary_classifier()
    binary_outputs = binary_model._forward_pass_fast(binary_X)
    assert np.allclose(
        mlp_classifier_probabilities_from_outputs(binary_outputs, n_outputs=binary_model.n_outputs_),
        binary_model.predict_proba(binary_X),
    )

    multiclass_model, multiclass_X, _ = _fit_multiclass_classifier()
    multiclass_outputs = multiclass_model._forward_pass_fast(multiclass_X)
    assert np.allclose(
        mlp_classifier_probabilities_from_outputs(multiclass_outputs, n_outputs=multiclass_model.n_outputs_),
        multiclass_model.predict_proba(multiclass_X),
    )


def test_mlp_classification_io_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_classification_io import (
        mlp_classifier_encode_targets,
        mlp_classifier_fit_target_state,
        mlp_classifier_labels_from_outputs,
        mlp_classifier_partial_fit_target_state,
        mlp_classifier_probabilities_from_outputs,
    )

    _, y = _binary_data()
    state = mlp_classifier_fit_target_state(y)

    with pytest.raises(ViolationError):
        mlp_classifier_fit_target_state(np.array([], dtype=np.int64))

    with pytest.raises(ViolationError):
        mlp_classifier_fit_target_state(y, incremental=True)

    with pytest.raises(ViolationError):
        mlp_classifier_partial_fit_target_state(y, np.array([[0, 1]], dtype=np.int64))

    with pytest.raises(ViolationError):
        mlp_classifier_encode_targets(np.array([], dtype=np.int64), state)

    with pytest.raises(ViolationError):
        mlp_classifier_labels_from_outputs(np.array([[np.nan]], dtype=np.float64), state, n_outputs=1)

    with pytest.raises(ViolationError):
        mlp_classifier_probabilities_from_outputs(np.array([[0.1]], dtype=np.float64), n_outputs=0)
