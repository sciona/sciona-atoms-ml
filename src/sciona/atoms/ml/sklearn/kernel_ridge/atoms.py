"""Kernel ridge atoms adapted from scikit-learn."""

from __future__ import annotations

import warnings

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from sklearn.metrics.pairwise import PAIRWISE_KERNEL_FUNCTIONS, pairwise_kernels
from sklearn.utils import check_array

from sciona.ghost.registry import register_atom

from .state_models import KernelRidgeState
from .witnesses import witness_kernel_ridge_fit, witness_kernel_ridge_predict


_SUPPORTED_KERNELS = frozenset(PAIRWISE_KERNEL_FUNCTIONS)
_NONNEGATIVE_KERNELS = {"additive_chi2", "chi2"}


def _matrix_2d(X: NDArray[np.float64]) -> bool:
    return bool(np.asarray(X).ndim == 2)


def _target_1d_or_2d(y: NDArray[np.float64]) -> bool:
    return bool(np.asarray(y).ndim in {1, 2})


def _same_sample_count(X: NDArray[np.float64], y: NDArray[np.float64]) -> bool:
    return bool(np.asarray(X).ndim == 2 and np.asarray(y).ndim in {1, 2} and np.asarray(X).shape[0] == np.asarray(y).shape[0])


def _finite_real(value: float) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)))


def _alpha_valid(alpha: float | tuple[float, ...] | NDArray[np.float64], y: NDArray[np.float64]) -> bool:
    values = np.atleast_1d(np.asarray(alpha, dtype=np.float64))
    n_outputs = 1 if np.asarray(y).ndim == 1 else np.asarray(y).shape[1]
    return bool(values.ndim == 1 and values.shape[0] in {1, n_outputs} and np.all(np.isfinite(values)) and np.all(values >= 0.0))


def _kernel_valid(kernel: str) -> bool:
    return bool(isinstance(kernel, str) and kernel in _SUPPORTED_KERNELS)


def _kernel_params_valid(kernel_params: dict[str, float] | None) -> bool:
    if kernel_params is None:
        return True
    return bool(
        isinstance(kernel_params, dict)
        and all(isinstance(key, str) and _finite_real(value) for key, value in kernel_params.items())
    )


def _values_valid_for_kernel(X: NDArray[np.float64], kernel: str) -> bool:
    values = np.asarray(X)
    return bool(values.ndim == 2 and (kernel not in _NONNEGATIVE_KERNELS or np.all(values >= 0.0)))


def _sample_weight_valid(sample_weight: float | tuple[float, ...] | NDArray[np.float64] | None, X: NDArray[np.float64]) -> bool:
    if sample_weight is None:
        return True
    values = np.atleast_1d(np.asarray(sample_weight, dtype=np.float64))
    return bool(
        values.ndim == 1
        and values.shape[0] in {1, np.asarray(X).shape[0]}
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
    )


def _state_valid(state: KernelRidgeState) -> bool:
    outputs = 1 if state.dual_coef.ndim == 1 else state.dual_coef.shape[1]
    return bool(
        state.X_fit.ndim == 2
        and state.X_fit.shape[1] == state.n_features_in
        and state.dual_coef.shape[0] == state.X_fit.shape[0]
        and state.alpha.ndim == 1
        and state.alpha.shape[0] in {1, outputs}
        and _kernel_valid(state.kernel)
        and (state.gamma is None or (_finite_real(state.gamma) and state.gamma >= 0.0))
        and _finite_real(state.degree)
        and state.degree >= 0.0
        and _finite_real(state.coef0)
        and np.all(np.isfinite(state.X_fit))
        and np.all(np.isfinite(state.dual_coef))
        and np.all(np.isfinite(state.alpha))
        and np.all(state.alpha >= 0.0)
    )


def _feature_count_matches(X: NDArray[np.float64], state: KernelRidgeState) -> bool:
    return bool(np.asarray(X).ndim == 2 and np.asarray(X).shape[1] == state.n_features_in)


def _prediction_valid(result: NDArray[np.float64], X: NDArray[np.float64], state: KernelRidgeState) -> bool:
    values = np.asarray(result)
    expected_shape = (np.asarray(X).shape[0],) if state.dual_coef.ndim == 1 else (np.asarray(X).shape[0], state.dual_coef.shape[1])
    return bool(values.shape == expected_shape and np.all(np.isfinite(values)))


def _kernel_params(gamma: float | None, degree: float, coef0: float) -> dict[str, float | None]:
    return {"gamma": gamma, "degree": float(degree), "coef0": float(coef0)}


def _solve_kernel_ridge(K: NDArray[np.float64], y: NDArray[np.float64], alpha: NDArray[np.float64], sample_weight: NDArray[np.float64] | None) -> NDArray[np.float64]:
    n_samples = K.shape[0]
    n_targets = y.shape[1]
    kernel_matrix = np.asarray(K, dtype=np.float64).copy()
    target = np.asarray(y, dtype=np.float64)
    weights = None
    if sample_weight is not None:
        weights = np.sqrt(np.atleast_1d(sample_weight))
        target = target * weights[:, np.newaxis]
        kernel_matrix *= np.outer(weights, weights)

    if np.all(alpha == alpha[0]):
        kernel_matrix.flat[:: n_samples + 1] += alpha[0]
        try:
            dual_coef = linalg.solve(kernel_matrix, target, assume_a="pos", overwrite_a=False)
        except np.linalg.LinAlgError:
            warnings.warn("Singular matrix in solving dual problem. Using least-squares solution instead.")
            dual_coef = linalg.lstsq(kernel_matrix, target)[0]
        if weights is not None:
            dual_coef *= weights[:, np.newaxis]
        return np.asarray(dual_coef, dtype=np.float64)

    dual_coefs = np.empty((n_targets, n_samples), dtype=np.float64)
    for dual_coef, target_column, current_alpha in zip(dual_coefs, target.T, alpha):
        kernel_matrix.flat[:: n_samples + 1] += current_alpha
        dual_coef[:] = linalg.solve(kernel_matrix, target_column, assume_a="pos", overwrite_a=False).ravel()
        kernel_matrix.flat[:: n_samples + 1] -= current_alpha
    if weights is not None:
        dual_coefs *= weights[np.newaxis, :]
    return np.asarray(dual_coefs.T, dtype=np.float64)


@register_atom(witness_kernel_ridge_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda y: _target_1d_or_2d(y), "y must be 1D or 2D")
@icontract.require(lambda X, y: _same_sample_count(X, y), "X and y must have matching sample counts")
@icontract.require(lambda alpha, y: _alpha_valid(alpha, y), "alpha must be non-negative and scalar or match output count")
@icontract.require(lambda kernel: _kernel_valid(kernel), "kernel must be a built-in pairwise kernel")
@icontract.require(lambda gamma: gamma is None or (_finite_real(gamma) and gamma >= 0.0), "gamma must be non-negative or None")
@icontract.require(lambda degree: _finite_real(degree) and degree >= 0.0, "degree must be non-negative")
@icontract.require(lambda coef0: _finite_real(coef0), "coef0 must be finite")
@icontract.require(lambda kernel_params: _kernel_params_valid(kernel_params), "kernel_params must contain finite numeric values")
@icontract.require(lambda X, kernel: _values_valid_for_kernel(X, kernel), "chi-square kernels require non-negative X")
@icontract.require(lambda sample_weight, X: _sample_weight_valid(sample_weight, X), "sample_weight must be non-negative and scalar or match sample count")
@icontract.ensure(lambda result: _state_valid(result), "kernel ridge state must contain finite dual coefficients")
def kernel_ridge_fit(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    alpha: float | tuple[float, ...] | NDArray[np.float64] = 1.0,
    kernel: str = "linear",
    gamma: float | None = None,
    degree: float = 3.0,
    coef0: float = 1.0,
    kernel_params: dict[str, float] | None = None,
    sample_weight: float | tuple[float, ...] | NDArray[np.float64] | None = None,
) -> KernelRidgeState:
    """Fit dense kernel ridge dual coefficients for built-in pairwise kernels."""
    del kernel_params
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    checked_y = check_array(y, dtype=np.float64, ensure_2d=False, input_name="y")
    ravel = checked_y.ndim == 1
    if ravel:
        checked_y = checked_y.reshape(-1, 1)
    alpha_values = np.atleast_1d(np.asarray(alpha, dtype=np.float64))
    weights = None if sample_weight is None else np.atleast_1d(np.asarray(sample_weight, dtype=np.float64))
    if weights is not None and weights.shape[0] == 1:
        weights = np.full(checked_x.shape[0], weights[0], dtype=np.float64)
    kernel_matrix = pairwise_kernels(checked_x, metric=kernel, filter_params=True, **_kernel_params(gamma, degree, coef0))
    dual_coef = _solve_kernel_ridge(kernel_matrix, checked_y, alpha_values, weights)
    if ravel:
        dual_coef = dual_coef.ravel()
    return KernelRidgeState(
        dual_coef=np.asarray(dual_coef, dtype=np.float64),
        X_fit=np.asarray(checked_x, dtype=np.float64).copy(),
        alpha=np.asarray(alpha_values, dtype=np.float64),
        kernel=kernel,
        gamma=gamma,
        degree=float(degree),
        coef0=float(coef0),
        n_features_in=int(checked_x.shape[1]),
    )


@register_atom(witness_kernel_ridge_predict)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X, state: _feature_count_matches(X, state), "X feature count must match fitted kernel ridge state")
@icontract.require(lambda X, state: _values_valid_for_kernel(X, state.kernel), "chi-square kernels require non-negative X")
@icontract.require(lambda state: _state_valid(state), "state must be a fitted kernel ridge state")
@icontract.ensure(lambda result, X, state: _prediction_valid(result, X, state), "predictions must match fitted output width")
def kernel_ridge_predict(X: NDArray[np.float64], state: KernelRidgeState) -> NDArray[np.float64]:
    """Predict dense outputs from fitted kernel ridge dual coefficients."""
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    kernel_matrix = pairwise_kernels(
        checked_x,
        state.X_fit,
        metric=state.kernel,
        filter_params=True,
        **_kernel_params(state.gamma, state.degree, state.coef0),
    )
    return np.asarray(np.dot(kernel_matrix, state.dual_coef), dtype=np.float64)
