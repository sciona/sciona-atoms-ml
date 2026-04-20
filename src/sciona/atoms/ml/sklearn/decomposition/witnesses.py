"""Ghost witnesses for selected sklearn decomposition atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray

from .state_models import TruncatedSVDState


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


def _check_2d(array: AbstractArray, name: str) -> tuple[int, int]:
    if len(array.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    return int(array.shape[0]), int(array.shape[1])
