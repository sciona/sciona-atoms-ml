"""Ghost witnesses for sklearn kernel ridge atoms."""

from __future__ import annotations

import math

from sklearn.metrics.pairwise import PAIRWISE_KERNEL_FUNCTIONS

from sciona.ghost.abstract import AbstractArray

from .state_models import KernelRidgeState


_SUPPORTED_KERNELS = frozenset(PAIRWISE_KERNEL_FUNCTIONS)


def _finite_real(value: float) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)))


def witness_kernel_ridge_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    alpha: float | tuple[float, ...] = 1.0,
    kernel: str = "linear",
    gamma: float | None = None,
    degree: float = 3.0,
    coef0: float = 1.0,
    kernel_params: dict[str, float] | None = None,
    sample_weight: float | tuple[float, ...] | None = None,
) -> AbstractArray:
    """Describe fitting dense kernel ridge dual coefficients."""
    del gamma, coef0, kernel_params, sample_weight
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if len(y.shape) not in {1, 2}:
        raise ValueError("y must be 1D or 2D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have matching sample counts")
    if not isinstance(kernel, str) or kernel not in _SUPPORTED_KERNELS:
        raise ValueError("kernel must be a built-in pairwise kernel name")
    if not _finite_real(degree) or degree < 0.0:
        raise ValueError("degree must be non-negative")
    alpha_values = alpha if isinstance(alpha, tuple) else (alpha,)
    if not all(_finite_real(value) and value >= 0.0 for value in alpha_values):
        raise ValueError("alpha must be non-negative")
    n_outputs = 1 if len(y.shape) == 1 else int(y.shape[1])
    if len(alpha_values) not in {1, n_outputs}:
        raise ValueError("alpha must be scalar or match output count")
    return AbstractArray(shape=(int(X.shape[0]), n_outputs), dtype="float64")


def witness_kernel_ridge_predict(X: AbstractArray, state: KernelRidgeState) -> AbstractArray:
    """Describe predicting with fitted kernel ridge dual coefficients."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    if len(state.dual_coef.shape) == 1:
        return AbstractArray(shape=(int(X.shape[0]),), dtype="float64")
    return AbstractArray(shape=(int(X.shape[0]), int(state.dual_coef.shape[1])), dtype="float64")
