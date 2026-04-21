"""Ghost witnesses for selected sklearn decomposition atoms."""

from __future__ import annotations

import numpy as np

from sciona.ghost.abstract import AbstractArray

from .state_models import FactorAnalysisState, FastICAState, IncrementalPCAState, KernelPCAState, TruncatedSVDState


def witness_pca_fit(
    X: AbstractArray,
    n_components: int | float | None = None,
    *,
    whiten: bool = False,
    copy: bool = True,
    svd_solver: str = "full",
) -> AbstractArray:
    """Describe fitting PCA components from a dense sample matrix."""
    del whiten, copy
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    n_samples, n_features = int(X.shape[0]), int(X.shape[1])
    if n_samples < 2:
        raise ValueError("PCA requires at least two samples")
    if svd_solver != "full":
        raise ValueError("this atom exposes the full-SVD PCA fit path")
    if n_components is None:
        width = min(n_samples, n_features)
    elif isinstance(n_components, int) and not isinstance(n_components, bool):
        if n_components < 0 or n_components > min(n_samples, n_features):
            raise ValueError("n_components must fit the full-SVD rank bound")
        width = n_components
    elif isinstance(n_components, float):
        if not 0.0 < n_components < 1.0:
            raise ValueError("fractional n_components must lie in (0, 1)")
        width = min(n_samples, n_features)
    else:
        raise ValueError("n_components must be None, an integer, or a float fraction")
    return AbstractArray(shape=(width, n_features), dtype="float64")


def witness_incremental_pca_partial_fit(
    X: AbstractArray,
    n_components: int | None = None,
    *,
    state: IncrementalPCAState | None = None,
    whiten: bool = False,
    copy: bool = True,
    batch_size: int | None = None,
) -> AbstractArray:
    """Describe updating incremental PCA components from one dense batch."""
    del whiten, copy, batch_size
    n_samples, n_features = _check_2d(X, "X")
    if state is None:
        if n_samples < 2:
            raise ValueError("first batch must contain at least two samples")
        if n_components is None:
            width = min(n_samples, n_features)
        elif isinstance(n_components, int) and not isinstance(n_components, bool):
            if n_components < 1 or n_components > min(n_samples, n_features):
                raise ValueError("n_components must fit the first batch")
            width = n_components
        else:
            raise ValueError("n_components must be None or a positive integer")
    else:
        if n_features != state.n_features_in:
            raise ValueError("X feature count must match fitted state")
        width = state.n_components if n_components is None else n_components
        if width != state.n_components:
            raise ValueError("n_components must match fitted state")
    return AbstractArray(shape=(int(width), n_features), dtype="float64")


def witness_incremental_pca_transform(X: AbstractArray, state: IncrementalPCAState) -> AbstractArray:
    """Describe projection onto incremental PCA components."""
    n_samples, n_features = _check_2d(X, "X")
    if n_features != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(n_samples, state.n_components), dtype="float64")


def witness_incremental_pca_inverse_transform(X: AbstractArray, state: IncrementalPCAState) -> AbstractArray:
    """Describe reconstruction from reduced coordinates."""
    n_samples, n_components = _check_2d(X, "X")
    if n_components != state.n_components:
        raise ValueError("X width must match fitted component count")
    return AbstractArray(shape=(n_samples, state.n_features_in), dtype="float64")


def witness_truncated_svd_fit(
    X: AbstractArray,
    n_components: int = 2,
    *,
    algorithm: str = "randomized",
    n_iter: int = 5,
    n_oversamples: int = 10,
    power_iteration_normalizer: str = "auto",
    random_state: int | None = None,
    tol: float = 0.0,
) -> AbstractArray:
    """Describe fitting randomized low-rank components."""
    del random_state
    n_samples, n_features = _check_2d(X, "X")
    if n_samples < 2 or n_features < 2:
        raise ValueError("X must have at least two samples and two features")
    if not isinstance(n_components, int) or isinstance(n_components, bool):
        raise ValueError("n_components must be a positive integer")
    if n_components < 1 or n_components > min(n_samples, n_features):
        raise ValueError("n_components must fit the dense randomized rank bound")
    if algorithm != "randomized":
        raise ValueError("this atom exposes the randomized truncated SVD path")
    if n_iter < 0:
        raise ValueError("n_iter must be non-negative")
    if n_oversamples < 1:
        raise ValueError("n_oversamples must be positive")
    if power_iteration_normalizer not in {"auto", "OR", "LU", "none"}:
        raise ValueError("unsupported power iteration normalizer")
    if tol < 0.0:
        raise ValueError("tol must be non-negative")
    return AbstractArray(shape=(n_components, n_features), dtype="float64")


def witness_truncated_svd_transform(X: AbstractArray, state: TruncatedSVDState) -> AbstractArray:
    """Describe projection onto fitted truncated SVD components."""
    n_samples, n_features = _check_2d(X, "X")
    if n_features != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(n_samples, state.n_components), dtype="float64")


def witness_truncated_svd_inverse_transform(X: AbstractArray, state: TruncatedSVDState) -> AbstractArray:
    """Describe reconstruction from truncated SVD component coordinates."""
    n_samples, n_components = _check_2d(X, "X")
    if n_components != state.n_components:
        raise ValueError("X width must match fitted component count")
    return AbstractArray(shape=(n_samples, state.n_features_in), dtype="float64")


def witness_kernel_pca_fit(
    X: AbstractArray,
    n_components: int = 2,
    *,
    kernel: str = "linear",
    gamma: float | None = None,
    degree: float = 3.0,
    coef0: float = 1.0,
    alpha: float = 1.0,
    fit_inverse_transform: bool = False,
    eigen_solver: str = "dense",
    tol: float = 0.0,
    max_iter: int | None = None,
    iterated_power: int | str = "auto",
    remove_zero_eig: bool = False,
    random_state: int | None = None,
    copy_X: bool = True,
    n_jobs: int | None = None,
) -> AbstractArray:
    """Describe fitting dense linear KernelPCA eigenvectors."""
    del gamma, copy_X, n_jobs, random_state
    n_samples, _ = _check_2d(X, "X")
    if n_samples < 2:
        raise ValueError("KernelPCA requires at least two samples")
    if not isinstance(n_components, int) or isinstance(n_components, bool) or n_components < 1:
        raise ValueError("n_components must be a positive integer")
    if kernel != "linear":
        raise ValueError("this atom exposes the linear kernel path")
    if degree < 0.0 or not np.isfinite(coef0) or alpha < 0.0:
        raise ValueError("kernel parameters must satisfy sklearn bounds")
    if fit_inverse_transform:
        raise ValueError("inverse transform fitting is outside this atom")
    if eigen_solver != "dense":
        raise ValueError("this atom exposes the dense eigensolver path")
    if tol < 0.0:
        raise ValueError("tol must be non-negative")
    if max_iter is not None and max_iter < 1:
        raise ValueError("max_iter must be positive when provided")
    if not (iterated_power == "auto" or (isinstance(iterated_power, int) and not isinstance(iterated_power, bool) and iterated_power >= 0)):
        raise ValueError("iterated_power must be 'auto' or a non-negative integer")
    if remove_zero_eig:
        raise ValueError("zero-eigenvalue removal is outside this fixed-width atom")
    return AbstractArray(shape=(n_samples, min(n_samples, n_components)), dtype="float64")


def witness_kernel_pca_transform(X: AbstractArray, state: KernelPCAState) -> AbstractArray:
    """Describe projection onto fitted kernel principal components."""
    n_samples, n_features = _check_2d(X, "X")
    if n_features != state.n_features_in:
        raise ValueError("X feature count must match fitted KernelPCA state")
    return AbstractArray(shape=(n_samples, state.n_components), dtype="float64")


def witness_factor_analysis_fit(
    X: AbstractArray,
    n_components: int | None = None,
    *,
    tol: float = 1e-2,
    copy: bool = True,
    max_iter: int = 1000,
    noise_variance_init: tuple[float, ...] | None = None,
    svd_method: str = "lapack",
    iterated_power: int = 3,
    rotation: None = None,
    random_state: int | None = 0,
) -> AbstractArray:
    """Describe fitting latent factors and noise variances."""
    del copy, noise_variance_init, iterated_power, random_state
    _, n_features = _check_2d(X, "X")
    if n_components is None:
        width = n_features
    elif isinstance(n_components, int) and not isinstance(n_components, bool):
        if n_components < 0 or n_components > n_features:
            raise ValueError("n_components must fit the feature count")
        width = n_components
    else:
        raise ValueError("n_components must be None or an integer")
    if tol < 0.0:
        raise ValueError("tol must be non-negative")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if svd_method != "lapack":
        raise ValueError("this atom exposes the Lapack factor analysis path")
    if rotation is not None:
        raise ValueError("rotation is outside this atom")
    return AbstractArray(shape=(width, n_features), dtype="float64")


def witness_factor_analysis_transform(X: AbstractArray, state: FactorAnalysisState) -> AbstractArray:
    """Describe expected latent factor means for samples."""
    n_samples, n_features = _check_2d(X, "X")
    if n_features != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(n_samples, state.n_components), dtype="float64")


def witness_factor_analysis_covariance(state: FactorAnalysisState) -> AbstractArray:
    """Describe covariance implied by factors and noise."""
    return AbstractArray(shape=(state.n_features_in, state.n_features_in), dtype="float64")


def witness_factor_analysis_precision(state: FactorAnalysisState) -> AbstractArray:
    """Describe inverse covariance implied by fitted factors."""
    return AbstractArray(shape=(state.n_features_in, state.n_features_in), dtype="float64")


def witness_factor_analysis_score_samples(X: AbstractArray, state: FactorAnalysisState) -> AbstractArray:
    """Describe per-sample log likelihood values."""
    n_samples, n_features = _check_2d(X, "X")
    if n_features != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(n_samples,), dtype="float64")


def witness_factor_analysis_score(X: AbstractArray, state: FactorAnalysisState) -> AbstractArray:
    """Describe average sample log likelihood."""
    _ = witness_factor_analysis_score_samples(X, state)
    return AbstractArray(shape=(), dtype="float64")


def witness_fastica_fit(
    X: AbstractArray,
    n_components: int | None = None,
    *,
    algorithm: str = "parallel",
    whiten: str | bool = "unit-variance",
    fun: str = "logcosh",
    fun_args: dict[str, float] | None = None,
    max_iter: int = 200,
    tol: float = 1e-4,
    w_init: tuple[tuple[float, ...], ...] | None = None,
    whiten_solver: str = "svd",
    random_state: int | None = None,
) -> AbstractArray:
    """Describe fitting a dense FastICA unmixing state."""
    del fun_args, w_init, random_state
    n_samples, n_features = _check_2d(X, "X")
    if n_samples < 2:
        raise ValueError("FastICA requires at least two samples")
    if whiten not in {"unit-variance", "arbitrary-variance", False}:
        raise ValueError("unsupported whitening mode")
    if n_components is None or whiten is False:
        width = min(n_samples, n_features)
    elif isinstance(n_components, int) and not isinstance(n_components, bool):
        if n_components < 1 or n_components > min(n_samples, n_features):
            raise ValueError("n_components must fit the sample/feature rank bound")
        width = n_components
    else:
        raise ValueError("n_components must be None or a positive integer")
    if algorithm not in {"parallel", "deflation"}:
        raise ValueError("unsupported FastICA algorithm")
    if fun not in {"logcosh", "exp", "cube"}:
        raise ValueError("unsupported FastICA nonlinearity")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if tol < 0.0:
        raise ValueError("tol must be non-negative")
    if whiten_solver not in {"svd", "eigh"}:
        raise ValueError("unsupported whitening solver")
    return AbstractArray(shape=(width, n_features), dtype="float64")


def witness_fastica_transform(X: AbstractArray, state: FastICAState) -> AbstractArray:
    """Describe recovering independent sources with fitted FastICA components."""
    n_samples, n_features = _check_2d(X, "X")
    if n_features != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(n_samples, state.n_components), dtype="float64")


def witness_fastica_inverse_transform(X: AbstractArray, state: FastICAState) -> AbstractArray:
    """Describe mapping FastICA source coordinates back to feature space."""
    n_samples, n_components = _check_2d(X, "X")
    if n_components != state.n_components:
        raise ValueError("X width must match fitted component count")
    return AbstractArray(shape=(n_samples, state.n_features_in), dtype="float64")


def _check_2d(array: AbstractArray, name: str) -> tuple[int, int]:
    if len(array.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    return int(array.shape[0]), int(array.shape[1])
