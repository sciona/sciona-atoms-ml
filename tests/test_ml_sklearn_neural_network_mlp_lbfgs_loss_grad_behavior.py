from __future__ import annotations

import numpy as np
import pytest
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network._multilayer_perceptron import _pack


def test_mlp_lbfgs_loss_grad_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_lbfgs_loss_grad import (
        mlp_lbfgs_loss_grad,
        mlp_lbfgs_unpack_parameters,
    )

    assert callable(mlp_lbfgs_unpack_parameters)
    assert callable(mlp_lbfgs_loss_grad)


def test_mlp_lbfgs_unpack_parameters_matches_source_layout() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_lbfgs_bookkeeping import (
        mlp_lbfgs_coef_indptr,
        mlp_lbfgs_intercept_indptr,
    )
    from sciona.atoms.ml.sklearn.neural_network.mlp_lbfgs_loss_grad import (
        mlp_lbfgs_unpack_parameters,
    )

    layer_units = (2, 3, 1)
    coef_indptr = mlp_lbfgs_coef_indptr(layer_units)
    intercept_indptr = mlp_lbfgs_intercept_indptr(layer_units, coef_indptr)
    packed = np.arange(13, dtype=np.float64)

    coefs, intercepts = mlp_lbfgs_unpack_parameters(packed, coef_indptr, intercept_indptr)

    assert len(coefs) == 2
    assert len(intercepts) == 2
    assert np.array_equal(coefs[0], packed[:6].reshape(2, 3))
    assert np.array_equal(coefs[1], packed[6:9].reshape(3, 1))
    assert np.array_equal(intercepts[0], packed[9:12])
    assert np.array_equal(intercepts[1], packed[12:13])


def test_mlp_lbfgs_loss_grad_matches_private_loss_grad_lbfgs() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_lbfgs_bookkeeping import (
        mlp_lbfgs_coef_indptr,
        mlp_lbfgs_intercept_indptr,
    )
    from sciona.atoms.ml.sklearn.neural_network.mlp_lbfgs_loss_grad import (
        mlp_lbfgs_loss_grad,
    )

    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    y = np.array([0.0, 1.0, 1.0, 2.0], dtype=np.float64)

    model = MLPRegressor(
        hidden_layer_sizes=(2,),
        activation="tanh",
        solver="lbfgs",
        alpha=0.25,
        random_state=0,
        max_iter=20,
    )
    model.fit(X, y)

    packed = _pack(model.coefs_, model.intercepts_)
    layer_units = (2, 2, 1)
    coef_indptr = mlp_lbfgs_coef_indptr(layer_units)
    intercept_indptr = mlp_lbfgs_intercept_indptr(layer_units, coef_indptr)
    y_2d = y.reshape(-1, 1)

    activations = [X] + [np.empty((X.shape[0], width), dtype=np.float64) for width in layer_units[1:]]
    deltas = [np.empty((X.shape[0], width), dtype=np.float64) for width in layer_units[1:]]
    coef_grads = [np.empty_like(coef) for coef in model.coefs_]
    intercept_grads = [np.empty_like(intercept) for intercept in model.intercepts_]

    expected_loss, expected_grad = model._loss_grad_lbfgs(
        packed.copy(),
        X,
        y_2d,
        activations,
        deltas,
        coef_grads,
        intercept_grads,
    )

    actual_loss, actual_grad = mlp_lbfgs_loss_grad(
        packed,
        coef_indptr,
        intercept_indptr,
        X,
        y_2d,
        hidden_activation="tanh",
        output_activation="identity",
        loss_name="squared_error",
        alpha=model.alpha,
    )

    assert actual_loss == pytest.approx(expected_loss)
    assert np.allclose(actual_grad, expected_grad)


def test_mlp_lbfgs_loss_grad_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_lbfgs_bookkeeping import (
        mlp_lbfgs_coef_indptr,
        mlp_lbfgs_intercept_indptr,
    )
    from sciona.atoms.ml.sklearn.neural_network.mlp_lbfgs_loss_grad import (
        mlp_lbfgs_loss_grad,
        mlp_lbfgs_unpack_parameters,
    )

    layer_units = (2, 1)
    coef_indptr = mlp_lbfgs_coef_indptr(layer_units)
    intercept_indptr = mlp_lbfgs_intercept_indptr(layer_units, coef_indptr)
    packed = np.zeros(4, dtype=np.float64)

    with pytest.raises(Exception):
        mlp_lbfgs_unpack_parameters(np.zeros(2, dtype=np.float64), coef_indptr, intercept_indptr)
    with pytest.raises(Exception):
        mlp_lbfgs_loss_grad(
            packed,
            coef_indptr,
            intercept_indptr,
            np.zeros((2, 2), dtype=np.float64),
            np.zeros((2, 2), dtype=np.float64),
            hidden_activation="relu",
            output_activation="identity",
            loss_name="squared_error",
        )
