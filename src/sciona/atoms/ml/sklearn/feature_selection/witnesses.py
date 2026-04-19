"""Ghost witnesses for sklearn feature-selection score atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_xy(X: AbstractArray, y: AbstractArray) -> tuple[int, int]:
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if len(y.shape) != 1:
        raise ValueError("y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have equal sample count")
    return int(X.shape[0]), int(X.shape[1])


def witness_f_classif(X: AbstractArray, y: AbstractArray) -> tuple[AbstractArray, AbstractArray]:
    """Describe ANOVA score and p-value vectors for class labels."""
    _n_samples, n_features = _check_xy(X, y)
    scores = AbstractArray(shape=(n_features,), dtype="float64", min_val=0.0)
    p_values = AbstractArray(shape=(n_features,), dtype="float64", min_val=0.0, max_val=1.0)
    return scores, p_values


def witness_chi2(X: AbstractArray, y: AbstractArray) -> tuple[AbstractArray, AbstractArray]:
    """Describe chi-square score and p-value vectors for class labels."""
    _n_samples, n_features = _check_xy(X, y)
    scores = AbstractArray(shape=(n_features,), dtype="float64", min_val=0.0)
    p_values = AbstractArray(shape=(n_features,), dtype="float64", min_val=0.0, max_val=1.0)
    return scores, p_values


def witness_r_regression(
    X: AbstractArray,
    y: AbstractArray,
    *,
    center: bool = True,
    force_finite: bool = True,
) -> AbstractArray:
    """Describe one Pearson correlation value per input feature."""
    del center, force_finite
    _n_samples, n_features = _check_xy(X, y)
    return AbstractArray(shape=(n_features,), dtype="float64")


def witness_f_regression(
    X: AbstractArray,
    y: AbstractArray,
    *,
    center: bool = True,
    force_finite: bool = True,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe regression F-score and p-value vectors per feature."""
    del center, force_finite
    _n_samples, n_features = _check_xy(X, y)
    scores = AbstractArray(shape=(n_features,), dtype="float64", min_val=0.0)
    p_values = AbstractArray(shape=(n_features,), dtype="float64", min_val=0.0, max_val=1.0)
    return scores, p_values
