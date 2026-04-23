from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state


def test_mlp_initialization_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_initialization import (
        mlp_glorot_init_bound,
        mlp_init_layer_parameters,
        mlp_initialize_parameters,
        mlp_output_activation_name,
    )

    assert callable(mlp_output_activation_name)
    assert callable(mlp_glorot_init_bound)
    assert callable(mlp_init_layer_parameters)
    assert callable(mlp_initialize_parameters)


def test_mlp_output_activation_name_matches_sklearn_branches() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_initialization import mlp_output_activation_name

    assert mlp_output_activation_name(is_classifier=True, label_binarizer_type="binary") == "logistic"
    assert mlp_output_activation_name(is_classifier=True, label_binarizer_type="multilabel-indicator") == "logistic"
    assert mlp_output_activation_name(is_classifier=True, label_binarizer_type="multiclass") == "softmax"
    assert mlp_output_activation_name(is_classifier=False, loss_name="squared_error") == "identity"
    assert mlp_output_activation_name(is_classifier=False, loss_name="poisson") == "exp"


def test_mlp_glorot_init_bound_matches_formula() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_initialization import mlp_glorot_init_bound

    assert mlp_glorot_init_bound(4, 3, activation="logistic") == pytest.approx(np.sqrt(2.0 / 7.0))
    assert mlp_glorot_init_bound(4, 3, activation="relu") == pytest.approx(np.sqrt(6.0 / 7.0))


def test_mlp_init_layer_parameters_matches_private_init_coef_float64() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_initialization import mlp_init_layer_parameters

    model = MLPClassifier(hidden_layer_sizes=(3,), activation="tanh", random_state=7)
    model._random_state = check_random_state(7)
    expected_coef, expected_intercept = model._init_coef(2, 3, np.float64)

    coef, intercept = mlp_init_layer_parameters(2, 3, activation="tanh", random_state=7, dtype_name="float64")

    assert np.array_equal(coef, expected_coef)
    assert np.array_equal(intercept, expected_intercept)


def test_mlp_init_layer_parameters_matches_private_init_coef_float32() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_initialization import mlp_init_layer_parameters

    model = MLPRegressor(hidden_layer_sizes=(3,), activation="logistic", random_state=11)
    model._random_state = check_random_state(11)
    expected_coef, expected_intercept = model._init_coef(2, 3, np.float32)

    coef, intercept = mlp_init_layer_parameters(2, 3, activation="logistic", random_state=11, dtype_name="float32")

    assert np.array_equal(coef, expected_coef)
    assert np.array_equal(intercept, expected_intercept)


def test_mlp_initialize_parameters_matches_private_initialize_sequence() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_initialization import mlp_initialize_parameters

    model = MLPClassifier(
        hidden_layer_sizes=(3,),
        activation="tanh",
        solver="sgd",
        early_stopping=True,
        random_state=13,
    )
    y = np.array([0, 1, 1, 0], dtype=np.int64)
    model._random_state = check_random_state(13)
    model._label_binarizer = LabelBinarizer().fit(y)
    y_encoded = model._label_binarizer.transform(y).astype(np.float64)
    model._initialize(y_encoded, [2, 3, 1], np.float64)

    coefs, intercepts, best_coefs, best_intercepts = mlp_initialize_parameters(
        (2, 3, 1),
        activation="tanh",
        random_state=13,
        dtype_name="float64",
    )

    for actual, target in zip(coefs, model.coefs_):
        assert np.array_equal(actual, target)
    for actual, target in zip(intercepts, model.intercepts_):
        assert np.array_equal(actual, target)
    for actual, target in zip(best_coefs, model._best_coefs):
        assert np.array_equal(actual, target)
    for actual, target in zip(best_intercepts, model._best_intercepts):
        assert np.array_equal(actual, target)


def test_mlp_initialization_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_initialization import (
        mlp_glorot_init_bound,
        mlp_init_layer_parameters,
        mlp_initialize_parameters,
        mlp_output_activation_name,
    )

    with pytest.raises(ViolationError):
        mlp_output_activation_name(is_classifier=True, label_binarizer_type=None)

    with pytest.raises(ViolationError):
        mlp_glorot_init_bound(0, 3, activation="relu")

    with pytest.raises(ViolationError):
        mlp_init_layer_parameters(2, 3, activation="relu", dtype_name="float16")  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        mlp_initialize_parameters((2,), activation="tanh")
