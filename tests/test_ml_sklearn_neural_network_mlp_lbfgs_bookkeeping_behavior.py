from __future__ import annotations

import numpy as np
import pytest
from sklearn.neural_network._multilayer_perceptron import _pack


def test_mlp_lbfgs_bookkeeping_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_lbfgs_bookkeeping import (
        mlp_lbfgs_coef_indptr,
        mlp_lbfgs_intercept_indptr,
        mlp_lbfgs_iprint,
        mlp_lbfgs_pack_parameters,
    )

    assert callable(mlp_lbfgs_coef_indptr)
    assert callable(mlp_lbfgs_intercept_indptr)
    assert callable(mlp_lbfgs_pack_parameters)
    assert callable(mlp_lbfgs_iprint)


def test_mlp_lbfgs_slice_layout_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_lbfgs_bookkeeping import (
        mlp_lbfgs_coef_indptr,
        mlp_lbfgs_intercept_indptr,
    )

    layer_units = (3, 5, 2)
    coef_indptr = mlp_lbfgs_coef_indptr(layer_units)
    intercept_indptr = mlp_lbfgs_intercept_indptr(layer_units, coef_indptr)

    assert coef_indptr == (
        (0, 15, (3, 5)),
        (15, 25, (5, 2)),
    )
    assert intercept_indptr == (
        (25, 30),
        (30, 32),
    )


def test_mlp_lbfgs_pack_parameters_matches_private_pack() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_lbfgs_bookkeeping import (
        mlp_lbfgs_pack_parameters,
    )

    coefs = (
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        np.array([[5.0], [6.0]], dtype=np.float64),
    )
    intercepts = (
        np.array([7.0, 8.0], dtype=np.float64),
        np.array([9.0], dtype=np.float64),
    )
    actual = mlp_lbfgs_pack_parameters(coefs, intercepts)
    expected = _pack(list(coefs), list(intercepts))
    assert np.array_equal(actual, expected)


def test_mlp_lbfgs_iprint_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_lbfgs_bookkeeping import mlp_lbfgs_iprint

    assert mlp_lbfgs_iprint(False) == -1
    assert mlp_lbfgs_iprint(0) == -1
    assert mlp_lbfgs_iprint(True) == 1
    assert mlp_lbfgs_iprint(2) == 1


def test_mlp_lbfgs_bookkeeping_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_lbfgs_bookkeeping import (
        mlp_lbfgs_coef_indptr,
        mlp_lbfgs_intercept_indptr,
        mlp_lbfgs_iprint,
        mlp_lbfgs_pack_parameters,
    )

    with pytest.raises(Exception):
        mlp_lbfgs_coef_indptr((3,))
    with pytest.raises(Exception):
        mlp_lbfgs_intercept_indptr((3, 2), ((0, 4, (2, 2)),))
    with pytest.raises(Exception):
        mlp_lbfgs_pack_parameters((np.array([1.0, 2.0]),), (np.array([1.0]),))
    with pytest.raises(Exception):
        mlp_lbfgs_iprint(np.bool_(True))
