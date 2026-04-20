"""Ghost witnesses for sklearn kernel approximation atoms."""

from __future__ import annotations

import math

from sklearn.metrics.pairwise import PAIRWISE_KERNEL_FUNCTIONS

from sciona.ghost.abstract import AbstractArray

from .state_models import NystroemState, PolynomialCountSketchState, RBFSamplerState, SkewedChi2SamplerState


_SUPPORTED_NYSTROEM_KERNELS = frozenset(PAIRWISE_KERNEL_FUNCTIONS)


def _finite_real(value: float) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)))


def witness_rbf_sampler_fit(
    X: AbstractArray,
    *,
    gamma: float | str = 1.0,
    n_components: int = 100,
    random_state: int | None = None,
) -> AbstractArray:
    """Describe fitting random Fourier weights for RBF features."""
    del random_state
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if not (gamma == "scale" or (isinstance(gamma, (int, float)) and not isinstance(gamma, bool) and gamma > 0.0)):
        raise ValueError("gamma must be positive or 'scale'")
    if not isinstance(n_components, int) or isinstance(n_components, bool) or n_components < 1:
        raise ValueError("n_components must be a positive integer")
    return AbstractArray(shape=(int(X.shape[1]), n_components), dtype="float64")


def witness_rbf_sampler_transform(X: AbstractArray, state: RBFSamplerState) -> AbstractArray:
    """Describe projecting samples into fitted RBF random features."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(int(X.shape[0]), state.n_components), dtype="float64")


def witness_skewed_chi2_sampler_fit(
    X: AbstractArray,
    *,
    skewedness: float = 1.0,
    n_components: int = 100,
    random_state: int | None = None,
) -> AbstractArray:
    """Describe fitting random Fourier weights for skewed chi-square features."""
    del random_state
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if not isinstance(skewedness, (int, float)) or isinstance(skewedness, bool):
        raise ValueError("skewedness must be finite")
    if not isinstance(n_components, int) or isinstance(n_components, bool) or n_components < 1:
        raise ValueError("n_components must be a positive integer")
    return AbstractArray(shape=(int(X.shape[1]), n_components), dtype="float64")


def witness_skewed_chi2_sampler_transform(X: AbstractArray, state: SkewedChi2SamplerState) -> AbstractArray:
    """Describe projecting samples into fitted skewed chi-square features."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(int(X.shape[0]), state.n_components), dtype="float64")


def witness_additive_chi2_sampler_transform(
    X: AbstractArray,
    *,
    sample_steps: int = 2,
    sample_interval: float | None = None,
) -> AbstractArray:
    """Describe explicit additive chi-square feature expansion."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if not isinstance(sample_steps, int) or isinstance(sample_steps, bool) or sample_steps < 1:
        raise ValueError("sample_steps must be positive")
    if sample_interval is None and sample_steps not in {1, 2, 3}:
        raise ValueError("sample_interval is required for this sample_steps value")
    if sample_interval is not None and sample_interval <= 0.0:
        raise ValueError("sample_interval must be positive")
    return AbstractArray(shape=(int(X.shape[0]), int(X.shape[1]) * (2 * sample_steps - 1)), dtype="float64")


def witness_polynomial_count_sketch_fit(
    X: AbstractArray,
    *,
    gamma: float = 1.0,
    degree: int = 2,
    coef0: float = 0.0,
    n_components: int = 100,
    random_state: int | None = None,
) -> AbstractArray:
    """Describe fitting Tensor Sketch hash tables for polynomial features."""
    del random_state
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    if not isinstance(degree, int) or isinstance(degree, bool) or degree < 1:
        raise ValueError("degree must be a positive integer")
    if coef0 < 0.0:
        raise ValueError("coef0 must be non-negative")
    if not isinstance(n_components, int) or isinstance(n_components, bool) or n_components < 1:
        raise ValueError("n_components must be a positive integer")
    width = int(X.shape[1]) + (1 if coef0 != 0.0 else 0)
    return AbstractArray(shape=(degree, width), dtype="int64")


def witness_polynomial_count_sketch_transform(X: AbstractArray, state: PolynomialCountSketchState) -> AbstractArray:
    """Describe projecting samples into polynomial count-sketch features."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(int(X.shape[0]), state.n_components), dtype="float64")


def witness_nystroem_fit(
    X: AbstractArray,
    *,
    kernel: str = "rbf",
    gamma: float | None = None,
    coef0: float | None = None,
    degree: float | None = None,
    kernel_params: dict[str, float] | None = None,
    n_components: int = 100,
    random_state: int | None = None,
    n_jobs: int | None = None,
) -> AbstractArray:
    """Describe fitting a dense Nystroem basis and normalization matrix."""
    del random_state
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if not isinstance(kernel, str) or kernel not in _SUPPORTED_NYSTROEM_KERNELS:
        raise ValueError("kernel must be a built-in pairwise kernel name")
    if gamma is not None and (not _finite_real(gamma) or gamma < 0.0):
        raise ValueError("gamma must be non-negative or None")
    if coef0 is not None and not _finite_real(coef0):
        raise ValueError("coef0 must be finite or None")
    if degree is not None and (not _finite_real(degree) or degree < 1.0):
        raise ValueError("degree must be at least one or None")
    if kernel_params is not None and (
        not isinstance(kernel_params, dict)
        or not all(isinstance(key, str) and _finite_real(value) for key, value in kernel_params.items())
    ):
        raise ValueError("kernel_params must contain finite numeric values")
    if not isinstance(n_components, int) or isinstance(n_components, bool) or n_components < 1:
        raise ValueError("n_components must be a positive integer")
    if n_jobs is not None and (not isinstance(n_jobs, int) or isinstance(n_jobs, bool)):
        raise ValueError("n_jobs must be None or an integer")
    width = min(int(X.shape[0]), n_components)
    return AbstractArray(shape=(width, int(X.shape[1])), dtype="float64")


def witness_nystroem_transform(X: AbstractArray, state: NystroemState) -> AbstractArray:
    """Describe projecting samples with a fitted dense Nystroem state."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(int(X.shape[0]), state.n_components), dtype="float64")
