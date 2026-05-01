from __future__ import annotations

import numpy as np
import pytest


def test_mlp_fit_buffer_setup_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_fit_buffer_setup import (
        mlp_fit_coef_gradient_buffers,
        mlp_fit_intercept_gradient_buffers,
        mlp_fit_layer_units,
        mlp_fit_targets_2d,
    )

    assert callable(mlp_fit_targets_2d)
    assert callable(mlp_fit_layer_units)
    assert callable(mlp_fit_coef_gradient_buffers)
    assert callable(mlp_fit_intercept_gradient_buffers)


def test_mlp_fit_targets_2d_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_fit_buffer_setup import (
        mlp_fit_targets_2d,
    )

    y_1d = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y_2d = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)

    assert np.array_equal(mlp_fit_targets_2d(y_1d), y_2d)
    assert np.array_equal(mlp_fit_targets_2d(y_2d), y_2d)


def test_mlp_fit_layer_units_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_fit_buffer_setup import (
        mlp_fit_layer_units,
    )

    assert mlp_fit_layer_units(4, (5, 3), 2) == (4, 5, 3, 2)


def test_mlp_fit_gradient_buffers_match_source_shapes_and_dtype() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_fit_buffer_setup import (
        mlp_fit_coef_gradient_buffers,
        mlp_fit_intercept_gradient_buffers,
        mlp_fit_layer_units,
    )

    layer_units = mlp_fit_layer_units(4, (5, 3), 2)
    coef_grads = mlp_fit_coef_gradient_buffers(layer_units, "float32")
    intercept_grads = mlp_fit_intercept_gradient_buffers(layer_units, "float32")

    assert [buffer.shape for buffer in coef_grads] == [(4, 5), (5, 3), (3, 2)]
    assert [buffer.shape for buffer in intercept_grads] == [(5,), (3,), (2,)]
    assert all(buffer.dtype == np.float32 for buffer in coef_grads)
    assert all(buffer.dtype == np.float32 for buffer in intercept_grads)


def test_mlp_fit_buffer_setup_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_fit_buffer_setup import (
        mlp_fit_coef_gradient_buffers,
        mlp_fit_intercept_gradient_buffers,
        mlp_fit_layer_units,
        mlp_fit_targets_2d,
    )

    with pytest.raises(Exception):
        mlp_fit_targets_2d(np.array([[1.0, np.inf]]))
    with pytest.raises(Exception):
        mlp_fit_layer_units(0, (2,), 1)
    with pytest.raises(Exception):
        mlp_fit_coef_gradient_buffers((3,), "float64")
    with pytest.raises(Exception):
        mlp_fit_intercept_gradient_buffers((3, 2), "not_a_dtype")
