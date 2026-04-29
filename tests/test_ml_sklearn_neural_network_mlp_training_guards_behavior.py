from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError

from sciona.atoms.ml.sklearn.neural_network.mlp_training_guards import (
    mlp_fit_require_finite_weights,
    mlp_partial_fit_require_stochastic_solver,
)


def test_partial_fit_require_stochastic_solver_accepts_supported_solvers() -> None:
    assert mlp_partial_fit_require_stochastic_solver(solver="sgd") is True
    assert mlp_partial_fit_require_stochastic_solver(solver="adam") is True


def test_partial_fit_require_stochastic_solver_rejects_nonstochastic_solver() -> None:
    with pytest.raises(AttributeError, match="partial_fit is only available for stochastic optimizers. lbfgs is not stochastic."):
        mlp_partial_fit_require_stochastic_solver(solver="lbfgs")


def test_fit_require_finite_weights_accepts_finite_parameters() -> None:
    coefs = [
        np.array([[0.1, -0.3], [0.5, 0.2]], dtype=np.float64),
        np.array([[0.9], [-0.4]], dtype=np.float64),
    ]
    intercepts = [
        np.array([0.0, 0.1], dtype=np.float64),
        np.array([0.3], dtype=np.float64),
    ]
    assert mlp_fit_require_finite_weights(coefs, intercepts) is True


def test_fit_require_finite_weights_rejects_nonfinite_parameters() -> None:
    coefs = [np.array([[0.1, np.inf]], dtype=np.float64)]
    intercepts = [np.array([0.0], dtype=np.float64)]
    with pytest.raises(ValueError, match="Solver produced non-finite parameter weights. The input data may contain large values and need to be preprocessed."):
        mlp_fit_require_finite_weights(coefs, intercepts)


def test_mlp_training_guards_reject_invalid_inputs() -> None:
    with pytest.raises((ViolationError, ValueError)):
        mlp_partial_fit_require_stochastic_solver(solver="")

    with pytest.raises((ViolationError, ValueError)):
        mlp_fit_require_finite_weights(
            [np.array([[1.0]], dtype=np.float64)],
            [np.array([1.0], dtype=np.float64), np.array([2.0], dtype=np.float64)],
        )
