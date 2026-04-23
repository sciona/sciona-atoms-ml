from __future__ import annotations

import warnings

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neural_network._base import (
    binary_log_loss as sklearn_binary_log_loss,
    inplace_identity,
    inplace_identity_derivative,
    inplace_logistic,
    inplace_logistic_derivative,
    inplace_relu,
    inplace_relu_derivative,
    inplace_softmax,
    inplace_tanh,
    inplace_tanh_derivative,
    log_loss as sklearn_log_loss,
    squared_loss as sklearn_squared_loss,
)


def _fit_binary_classifier() -> tuple[MLPClassifier, np.ndarray, np.ndarray]:
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
    y_encoded = model._label_binarizer.transform(y).astype(np.float64)
    return model, X, y_encoded


def _fit_multiclass_classifier() -> tuple[MLPClassifier, np.ndarray, np.ndarray]:
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
    y_encoded = model._label_binarizer.transform(y).astype(np.float64)
    return model, X, y_encoded


def _fit_regressor() -> tuple[MLPRegressor, np.ndarray, np.ndarray]:
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    y = np.array([0.1, 0.8, 0.75, 0.2], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = MLPRegressor(
            hidden_layer_sizes=(3,),
            activation="relu",
            solver="lbfgs",
            alpha=0.05,
            random_state=2,
            max_iter=80,
        ).fit(X, y)
    return model, X, y.reshape(-1, 1)


def _private_forward_pass(model: MLPClassifier | MLPRegressor, X: np.ndarray) -> list[np.ndarray]:
    activations = [X.copy()]
    for coef in model.coefs_:
        activations.append(np.empty((X.shape[0], coef.shape[1]), dtype=np.float64))
    return model._forward_pass(activations)


def _private_backprop(
    model: MLPClassifier | MLPRegressor,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> tuple[float, list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    activations = [X.copy()]
    for coef in model.coefs_:
        activations.append(np.empty((X.shape[0], coef.shape[1]), dtype=np.float64))
    deltas = [np.empty_like(layer) for layer in activations[1:]]
    coef_grads = [np.empty_like(coef) for coef in model.coefs_]
    intercept_grads = [np.empty_like(intercept) for intercept in model.intercepts_]
    loss, coef_grads, intercept_grads = model._backprop(
        X,
        y,
        sample_weight,
        activations,
        deltas,
        coef_grads,
        intercept_grads,
    )
    return loss, activations, deltas, coef_grads, intercept_grads


def test_mlp_primitives_import() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_primitives import (
        mlp_activation,
        mlp_activation_derivative,
        mlp_backprop,
        mlp_forward_pass,
        mlp_layer_gradients,
        mlp_loss,
    )

    assert callable(mlp_activation)
    assert callable(mlp_activation_derivative)
    assert callable(mlp_loss)
    assert callable(mlp_forward_pass)
    assert callable(mlp_layer_gradients)
    assert callable(mlp_backprop)


def test_mlp_activation_matches_sklearn_helpers() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_primitives import mlp_activation

    values = np.array([[1.0, -2.0, 0.5], [-0.25, 0.0, 3.0]], dtype=np.float64)
    helpers = {
        "identity": inplace_identity,
        "logistic": inplace_logistic,
        "tanh": inplace_tanh,
        "relu": inplace_relu,
        "softmax": inplace_softmax,
    }

    for name, helper in helpers.items():
        expected = values.copy()
        helper(expected)
        result = mlp_activation(values, activation=name)
        assert np.allclose(result, expected)


def test_mlp_activation_derivative_matches_sklearn_helpers() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_primitives import mlp_activation, mlp_activation_derivative

    pre_activation = np.array([[1.0, -2.0, 0.5], [-0.25, 0.0, 3.0]], dtype=np.float64)
    base_delta = np.array([[0.2, -0.4, 0.8], [1.0, -1.5, 0.3]], dtype=np.float64)
    derivative_helpers = {
        "identity": inplace_identity_derivative,
        "logistic": inplace_logistic_derivative,
        "tanh": inplace_tanh_derivative,
        "relu": inplace_relu_derivative,
    }

    for name, helper in derivative_helpers.items():
        activated = mlp_activation(pre_activation, activation=name)
        expected = base_delta.copy()
        helper(activated.copy(), expected)
        result = mlp_activation_derivative(activated, base_delta, activation=name)
        assert np.allclose(result, expected)


def test_mlp_loss_matches_sklearn_helpers() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_primitives import mlp_loss

    y_binary = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float64)
    p_binary = np.array([[0.1], [0.8], [0.7], [0.2]], dtype=np.float64)
    sample_weight = np.array([1.0, 2.0, 0.5, 1.5], dtype=np.float64)
    assert mlp_loss(y_binary, p_binary, loss_name="log_loss", output_activation="logistic") == pytest.approx(
        sklearn_binary_log_loss(y_binary, p_binary)
    )
    assert mlp_loss(
        y_binary,
        p_binary,
        loss_name="log_loss",
        output_activation="logistic",
        sample_weight=sample_weight,
    ) == pytest.approx(sklearn_binary_log_loss(y_binary, p_binary, sample_weight=sample_weight))

    y_multiclass = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    p_multiclass = np.array([[0.8, 0.1, 0.1], [0.2, 0.5, 0.3]], dtype=np.float64)
    assert mlp_loss(y_multiclass, p_multiclass, loss_name="log_loss", output_activation="softmax") == pytest.approx(
        sklearn_log_loss(y_multiclass, p_multiclass)
    )

    y_regression = np.array([[0.1], [0.8]], dtype=np.float64)
    p_regression = np.array([[0.2], [0.6]], dtype=np.float64)
    assert mlp_loss(
        y_regression,
        p_regression,
        loss_name="squared_error",
        output_activation="identity",
    ) == pytest.approx(sklearn_squared_loss(y_regression, p_regression))


def test_mlp_forward_pass_matches_private_method_for_binary_classifier() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_primitives import mlp_forward_pass

    model, X, _ = _fit_binary_classifier()
    expected = _private_forward_pass(model, X)
    result = mlp_forward_pass(
        X,
        tuple(model.coefs_),
        tuple(model.intercepts_),
        hidden_activation=model.activation,
        output_activation=model.out_activation_,
    )

    assert len(result) == len(expected)
    for actual, target in zip(result, expected):
        assert np.allclose(actual, target)


def test_mlp_forward_pass_matches_private_method_for_multiclass_classifier() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_primitives import mlp_forward_pass

    model, X, _ = _fit_multiclass_classifier()
    expected = _private_forward_pass(model, X)
    result = mlp_forward_pass(
        X,
        tuple(model.coefs_),
        tuple(model.intercepts_),
        hidden_activation=model.activation,
        output_activation=model.out_activation_,
    )

    assert len(result) == len(expected)
    for actual, target in zip(result, expected):
        assert np.allclose(actual, target)


def test_mlp_layer_gradients_match_private_helper() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_primitives import mlp_layer_gradients

    model, X, y = _fit_multiclass_classifier()
    _, activations, deltas, coef_grads, intercept_grads = _private_backprop(model, X, y)

    for index, coef in enumerate(model.coefs_):
        coef_grad, intercept_grad = mlp_layer_gradients(
            activations[index],
            deltas[index],
            coef,
            alpha=model.alpha,
            sample_weight_sum=float(X.shape[0]),
        )
        assert np.allclose(coef_grad, coef_grads[index])
        assert np.allclose(intercept_grad, intercept_grads[index])


def test_mlp_backprop_matches_private_method_for_binary_classifier() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_primitives import mlp_backprop

    model, X, y = _fit_binary_classifier()
    expected_loss, _, _, expected_coef_grads, expected_intercept_grads = _private_backprop(model, X, y)
    loss, coef_grads, intercept_grads = mlp_backprop(
        X,
        y,
        tuple(model.coefs_),
        tuple(model.intercepts_),
        hidden_activation=model.activation,
        output_activation=model.out_activation_,
        loss_name=model.loss,
        alpha=model.alpha,
    )

    assert loss == pytest.approx(expected_loss)
    for actual, target in zip(coef_grads, expected_coef_grads):
        assert np.allclose(actual, target)
    for actual, target in zip(intercept_grads, expected_intercept_grads):
        assert np.allclose(actual, target)


def test_mlp_backprop_matches_private_method_for_multiclass_classifier() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_primitives import mlp_backprop

    model, X, y = _fit_multiclass_classifier()
    expected_loss, _, _, expected_coef_grads, expected_intercept_grads = _private_backprop(model, X, y)
    loss, coef_grads, intercept_grads = mlp_backprop(
        X,
        y,
        tuple(model.coefs_),
        tuple(model.intercepts_),
        hidden_activation=model.activation,
        output_activation=model.out_activation_,
        loss_name=model.loss,
        alpha=model.alpha,
    )

    assert loss == pytest.approx(expected_loss)
    for actual, target in zip(coef_grads, expected_coef_grads):
        assert np.allclose(actual, target)
    for actual, target in zip(intercept_grads, expected_intercept_grads):
        assert np.allclose(actual, target)


def test_mlp_backprop_matches_private_method_for_regressor() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_primitives import mlp_backprop

    model, X, y = _fit_regressor()
    expected_loss, _, _, expected_coef_grads, expected_intercept_grads = _private_backprop(model, X, y)
    loss, coef_grads, intercept_grads = mlp_backprop(
        X,
        y,
        tuple(model.coefs_),
        tuple(model.intercepts_),
        hidden_activation=model.activation,
        output_activation=model.out_activation_,
        loss_name=model.loss,
        alpha=model.alpha,
    )

    assert loss == pytest.approx(expected_loss)
    for actual, target in zip(coef_grads, expected_coef_grads):
        assert np.allclose(actual, target)
    for actual, target in zip(intercept_grads, expected_intercept_grads):
        assert np.allclose(actual, target)


def test_mlp_backprop_matches_private_method_with_sample_weight() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_primitives import mlp_backprop

    model, X, y = _fit_binary_classifier()
    sample_weight = np.array([1.0, 2.0, 0.5, 1.5], dtype=np.float64)
    expected_loss, _, _, expected_coef_grads, expected_intercept_grads = _private_backprop(model, X, y, sample_weight)
    loss, coef_grads, intercept_grads = mlp_backprop(
        X,
        y,
        tuple(model.coefs_),
        tuple(model.intercepts_),
        hidden_activation=model.activation,
        output_activation=model.out_activation_,
        loss_name=model.loss,
        alpha=model.alpha,
        sample_weight=sample_weight,
    )

    assert loss == pytest.approx(expected_loss)
    for actual, target in zip(coef_grads, expected_coef_grads):
        assert np.allclose(actual, target)
    for actual, target in zip(intercept_grads, expected_intercept_grads):
        assert np.allclose(actual, target)


def test_mlp_primitives_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_primitives import (
        mlp_activation,
        mlp_backprop,
        mlp_forward_pass,
        mlp_loss,
    )

    with pytest.raises(ViolationError):
        mlp_activation(np.ones((2, 2), dtype=np.float64), activation="swish")  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        mlp_forward_pass(
            np.ones((2, 2), dtype=np.float64),
            (np.ones((3, 2), dtype=np.float64),),
            (np.zeros(2, dtype=np.float64),),
            hidden_activation="relu",
            output_activation="identity",
        )

    with pytest.raises(ViolationError):
        mlp_loss(
            np.ones((2, 1), dtype=np.float64),
            np.ones((2, 1), dtype=np.float64),
            loss_name="squared_error",
            output_activation="logistic",
        )

    with pytest.raises(ViolationError):
        mlp_backprop(
            np.ones((2, 2), dtype=np.float64),
            np.ones((2, 1), dtype=np.float64),
            (np.ones((2, 2), dtype=np.float64),),
            (np.zeros(2, dtype=np.float64),),
            hidden_activation="relu",
            output_activation="softmax",
            loss_name="log_loss",
        )

    with pytest.raises(ViolationError):
        mlp_backprop(
            np.ones((2, 2), dtype=np.float64),
            np.ones((2, 2), dtype=np.float64),
            (np.ones((2, 2), dtype=np.float64),),
            (np.zeros(2, dtype=np.float64),),
            hidden_activation="relu",
            output_activation="identity",
            loss_name="squared_error",
            sample_weight=np.array([1.0, -1.0], dtype=np.float64),
        )
