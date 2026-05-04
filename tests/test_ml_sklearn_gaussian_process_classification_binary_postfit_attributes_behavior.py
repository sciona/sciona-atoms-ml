from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.gaussian_process._gpc import _BinaryGaussianProcessClassifierLaplace
from sklearn.gaussian_process.kernels import ConstantKernel, RBF


def _fit_model() -> _BinaryGaussianProcessClassifierLaplace:
    X = np.array([[-1.0], [-0.3], [0.2], [0.9], [1.4]], dtype=np.float64)
    y = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    kernel = ConstantKernel(1.1, constant_value_bounds="fixed") * RBF(0.9, length_scale_bounds="fixed")
    model = _BinaryGaussianProcessClassifierLaplace(
        kernel=kernel,
        optimizer=None,
        max_iter_predict=25,
        warm_start=False,
    )
    model.fit(X, y)
    return model


def test_classification_binary_postfit_attributes_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_binary_postfit_attributes import (
        gpc_binary_fit_L,
        gpc_binary_fit_log_marginal_likelihood_value,
        gpc_binary_fit_pi,
        gpc_binary_fit_return_self,
        gpc_binary_fit_W_sr,
    )

    assert callable(gpc_binary_fit_pi)
    assert callable(gpc_binary_fit_W_sr)
    assert callable(gpc_binary_fit_L)
    assert callable(gpc_binary_fit_log_marginal_likelihood_value)
    assert callable(gpc_binary_fit_return_self)


def test_classification_binary_postfit_attributes_match_fitted_model() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_binary_postfit_attributes import (
        gpc_binary_fit_L,
        gpc_binary_fit_log_marginal_likelihood_value,
        gpc_binary_fit_pi,
        gpc_binary_fit_return_self,
        gpc_binary_fit_W_sr,
    )

    model = _fit_model()

    assert np.allclose(gpc_binary_fit_pi(model.pi_), model.pi_)
    assert np.allclose(gpc_binary_fit_W_sr(model.W_sr_), model.W_sr_)
    assert np.allclose(gpc_binary_fit_L(model.L_), model.L_)
    assert np.isclose(
        gpc_binary_fit_log_marginal_likelihood_value(model.log_marginal_likelihood_value_),
        model.log_marginal_likelihood_value_,
    )
    assert gpc_binary_fit_return_self("_BinaryGaussianProcessClassifierLaplace") == "_BinaryGaussianProcessClassifierLaplace"


def test_classification_binary_postfit_attributes_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_binary_postfit_attributes import (
        gpc_binary_fit_L,
        gpc_binary_fit_log_marginal_likelihood_value,
        gpc_binary_fit_pi,
        gpc_binary_fit_return_self,
        gpc_binary_fit_W_sr,
    )

    with pytest.raises(ViolationError):
        gpc_binary_fit_pi(np.array([0.0, 0.4], dtype=np.float64))

    with pytest.raises(ViolationError):
        gpc_binary_fit_W_sr(np.array([0.2, -0.1], dtype=np.float64))

    with pytest.raises(ViolationError):
        gpc_binary_fit_L(np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64))

    with pytest.raises(ViolationError):
        gpc_binary_fit_log_marginal_likelihood_value(float("nan"))

    with pytest.raises(ViolationError):
        gpc_binary_fit_return_self("")
