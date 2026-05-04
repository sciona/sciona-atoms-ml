from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from scipy.linalg import solve
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


def test_classification_binary_predict_proba_outputs_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_binary_predict_proba_outputs import (
        gpc_binary_predict_proba_alpha,
        gpc_binary_predict_proba_gamma,
        gpc_binary_predict_proba_integrals,
        gpc_binary_predict_proba_matrix,
        gpc_binary_predict_proba_positive_class_probabilities,
    )

    assert callable(gpc_binary_predict_proba_alpha)
    assert callable(gpc_binary_predict_proba_gamma)
    assert callable(gpc_binary_predict_proba_integrals)
    assert callable(gpc_binary_predict_proba_positive_class_probabilities)
    assert callable(gpc_binary_predict_proba_matrix)


def test_classification_binary_predict_proba_outputs_match_fitted_model() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_binary_predict_proba_outputs import (
        gpc_binary_predict_proba_alpha,
        gpc_binary_predict_proba_gamma,
        gpc_binary_predict_proba_integrals,
        gpc_binary_predict_proba_matrix,
        gpc_binary_predict_proba_positive_class_probabilities,
    )

    model = _fit_model()
    X_query = np.array([[-0.5], [0.15], [1.1]], dtype=np.float64)

    K_star = model.kernel_(model.X_train_, X_query)
    f_star = K_star.T.dot(model.y_train_ - model.pi_)
    v = solve(model.L_, model.W_sr_[:, np.newaxis] * K_star)
    var_f_star = model.kernel_.diag(X_query) - np.einsum("ij,ij->j", v, v)

    alpha = gpc_binary_predict_proba_alpha(var_f_star)
    gamma = gpc_binary_predict_proba_gamma(f_star)
    integrals = gpc_binary_predict_proba_integrals(alpha, gamma)
    pi_star = gpc_binary_predict_proba_positive_class_probabilities(integrals)
    probability_matrix = gpc_binary_predict_proba_matrix(pi_star)

    expected = model.predict_proba(X_query)
    assert np.allclose(probability_matrix, expected)
    assert np.allclose(pi_star, expected[:, 1])
    assert np.allclose(probability_matrix.sum(axis=1), 1.0)


def test_classification_binary_predict_proba_outputs_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_binary_predict_proba_outputs import (
        gpc_binary_predict_proba_alpha,
        gpc_binary_predict_proba_gamma,
        gpc_binary_predict_proba_integrals,
        gpc_binary_predict_proba_matrix,
        gpc_binary_predict_proba_positive_class_probabilities,
    )

    with pytest.raises(ViolationError):
        gpc_binary_predict_proba_alpha(np.array([0.0, 0.2], dtype=np.float64))

    with pytest.raises(ViolationError):
        gpc_binary_predict_proba_gamma(np.array([0.3, np.nan], dtype=np.float64))

    with pytest.raises(ViolationError):
        gpc_binary_predict_proba_integrals(
            np.array([0.5, 0.7], dtype=np.float64),
            np.ones((4, 2), dtype=np.float64),
        )

    with pytest.raises(ViolationError):
        gpc_binary_predict_proba_positive_class_probabilities(np.ones((4, 2), dtype=np.float64))

    with pytest.raises(ViolationError):
        gpc_binary_predict_proba_matrix(np.array([-0.1, 0.4], dtype=np.float64))
