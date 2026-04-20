from __future__ import annotations

import numpy as np
import pytest
from sklearn.kernel_ridge import KernelRidge as SklearnKernelRidge


def _data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 0.5],
            [3.0, 4.0, 1.5],
            [4.0, 2.0, 2.5],
        ],
        dtype=np.float64,
    )
    y = np.array([1.0, 2.0, 1.5, 3.5, 2.5], dtype=np.float64)
    return X, y


def test_kernel_ridge_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.kernel_ridge import KernelRidgeState, kernel_ridge_fit, kernel_ridge_predict

    assert KernelRidgeState is not None
    assert callable(kernel_ridge_fit)
    assert callable(kernel_ridge_predict)


def test_kernel_ridge_fit_and_predict_match_sklearn_linear() -> None:
    from sciona.atoms.ml.sklearn.kernel_ridge import kernel_ridge_fit, kernel_ridge_predict

    X, y = _data()
    state = kernel_ridge_fit(X, y, alpha=0.75, kernel="linear")
    expected = SklearnKernelRidge(alpha=0.75, kernel="linear").fit(X, y)

    assert np.allclose(state.dual_coef, expected.dual_coef_)
    assert np.allclose(state.X_fit, expected.X_fit_)
    assert state.n_features_in == expected.n_features_in_
    assert np.allclose(kernel_ridge_predict(X, state), expected.predict(X))


def test_kernel_ridge_fit_and_predict_match_sklearn_rbf_multioutput_weighted() -> None:
    from sciona.atoms.ml.sklearn.kernel_ridge import kernel_ridge_fit, kernel_ridge_predict

    X, y_1d = _data()
    y = np.column_stack([y_1d, y_1d**2])
    weights = np.array([1.0, 0.5, 1.5, 2.0, 0.75], dtype=np.float64)
    alpha = np.array([0.25, 0.5], dtype=np.float64)
    state = kernel_ridge_fit(X, y, alpha=alpha, kernel="rbf", gamma=0.2, sample_weight=weights)
    expected = SklearnKernelRidge(alpha=alpha, kernel="rbf", gamma=0.2).fit(X, y, sample_weight=weights)

    assert np.allclose(state.dual_coef, expected.dual_coef_)
    assert np.allclose(kernel_ridge_predict(X[:3], state), expected.predict(X[:3]))


def test_kernel_ridge_rejects_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.kernel_ridge import kernel_ridge_fit, kernel_ridge_predict

    X, y = _data()
    with pytest.raises(Exception):
        kernel_ridge_fit(X, y[:-1])
    with pytest.raises(Exception):
        kernel_ridge_fit(X, y, alpha=-0.1)
    with pytest.raises(Exception):
        kernel_ridge_fit(X, y, kernel="precomputed")
    with pytest.raises(Exception):
        kernel_ridge_fit(X, y, gamma=-0.1)
    with pytest.raises(Exception):
        kernel_ridge_fit(np.array([[1.0, -1.0]], dtype=np.float64), np.array([1.0]), kernel="chi2")
    with pytest.raises(Exception):
        kernel_ridge_fit(X, y, sample_weight=(1.0, 2.0))

    state = kernel_ridge_fit(X, y)
    with pytest.raises(Exception):
        kernel_ridge_predict(np.ones((2, 2), dtype=np.float64), state)
