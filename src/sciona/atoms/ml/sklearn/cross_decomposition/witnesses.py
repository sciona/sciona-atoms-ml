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


def _check_2d(array: AbstractArray, name: str) -> tuple[int, int]:
    if len(array.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    return int(array.shape[0]), int(array.shape[1])
