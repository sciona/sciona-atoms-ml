"""Ghost witnesses for sklearn cross-decomposition atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_plssvd_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    n_components: int = 2,
    scale: bool = True,
    copy: bool = True,
) -> AbstractArray:
    """Describe PLS-SVD fitting as feature weights for each component."""
    del scale, copy
    n_samples, n_features = _check_2d(X, "X")
    if len(y.shape) == 1:
        y_samples = y.shape[0]
        n_targets = 1
    else:
        y_samples, n_targets = _check_2d(y, "y")
    if y_samples != n_samples:
        raise ValueError("X and y must have matching sample counts")
    upper_bound = min(n_samples, n_features, n_targets)
    if n_components < 1 or n_components > upper_bound:
        raise ValueError("n_components must fit the cross-covariance rank bound")
    return AbstractArray(shape=(n_features, n_components), dtype="float64")


def witness_pls_regression_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    n_components: int = 2,
    scale: bool = True,
    max_iter: int = 500,
    tol: float = 1e-6,
    copy: bool = True,
) -> AbstractArray:
    """Describe fitting PLS regression state."""
    del scale, copy
    return _witness_pls_fit(X, y, n_components=n_components, max_iter=max_iter, tol=tol, regression_bound=True)


def witness_pls_canonical_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    n_components: int = 2,
    scale: bool = True,
    algorithm: str = "nipals",
    max_iter: int = 500,
    tol: float = 1e-6,
    copy: bool = True,
) -> AbstractArray:
    """Describe fitting PLS canonical state."""
    del scale, copy
    if algorithm not in {"nipals", "svd"}:
        raise ValueError("algorithm must be 'nipals' or 'svd'")
    return _witness_pls_fit(X, y, n_components=n_components, max_iter=max_iter, tol=tol, regression_bound=False)


def witness_cca_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    n_components: int = 2,
    scale: bool = True,
    max_iter: int = 500,
    tol: float = 1e-6,
    copy: bool = True,
) -> AbstractArray:
    """Describe fitting CCA state."""
    del scale, copy
    return _witness_pls_fit(X, y, n_components=n_components, max_iter=max_iter, tol=tol, regression_bound=False)


def _check_2d(array: AbstractArray, name: str) -> tuple[int, int]:
    if len(array.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    return int(array.shape[0]), int(array.shape[1])


def _witness_pls_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    n_components: int,
    max_iter: int,
    tol: float,
    regression_bound: bool,
) -> AbstractArray:
    n_samples, n_features = _check_2d(X, "X")
    if len(y.shape) == 1:
        y_samples = y.shape[0]
        n_targets = 1
    else:
        y_samples, n_targets = _check_2d(y, "y")
    if y_samples != n_samples:
        raise ValueError("X and y must have matching sample counts")
    upper_bound = min(n_samples, n_features) if regression_bound else min(n_samples, n_features, n_targets)
    if n_components < 1 or n_components > upper_bound:
        raise ValueError("n_components exceeds the PLS rank bound")
    if max_iter < 1:
        raise ValueError("max_iter must be at least one")
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    return AbstractArray(shape=(n_features, n_components), dtype="float64")
