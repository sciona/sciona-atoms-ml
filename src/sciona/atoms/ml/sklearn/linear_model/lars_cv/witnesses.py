"""Ghost witnesses for sklearn LARS cross-validation helper atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_lars_cv_residual_path(
    X_test: AbstractArray,
    y_test: AbstractArray,
    coefs: AbstractArray,
    x_mean: AbstractArray,
    *,
    y_mean: float = 0.0,
) -> AbstractArray:
    """Describe one left-out residual vector for each alpha in a supplied coefficient path."""
    del y_mean
    if len(X_test.shape) != 2 or len(coefs.shape) != 2:
        raise ValueError("X_test and coefs must be two-dimensional")
    if len(y_test.shape) != 1 or len(x_mean.shape) != 1:
        raise ValueError("y_test and x_mean must be one-dimensional")
    if X_test.shape[0] != y_test.shape[0] or X_test.shape[1] != coefs.shape[0] or X_test.shape[1] != x_mean.shape[0]:
        raise ValueError("test inputs must align with coefficient and mean dimensions")
    return AbstractArray(shape=(int(coefs.shape[1]), int(X_test.shape[0])), dtype="float64")


def witness_lars_cv_alpha_grid(
    fold_alphas: tuple[AbstractArray, ...],
    *,
    max_n_alphas: int = 1000,
) -> AbstractArray:
    """Describe a one-dimensional shared alpha grid built from fold alpha paths."""
    del max_n_alphas
    if not fold_alphas:
        raise ValueError("fold_alphas must be nonempty")
    for values in fold_alphas:
        if len(values.shape) != 1:
            raise ValueError("each fold alpha path must be one-dimensional")
    total = sum(int(values.shape[0]) for values in fold_alphas)
    return AbstractArray(shape=(total,), dtype="float64")


def witness_lars_cv_interpolated_fold_mse(
    fold_alphas: AbstractArray,
    fold_residues: AbstractArray,
    target_alphas: AbstractArray,
) -> AbstractArray:
    """Describe one projected MSE value for each alpha in the shared grid."""
    if len(fold_alphas.shape) != 1 or len(target_alphas.shape) != 1:
        raise ValueError("fold_alphas and target_alphas must be one-dimensional")
    if len(fold_residues.shape) != 2 or fold_residues.shape[0] != fold_alphas.shape[0]:
        raise ValueError("fold_residues must be a two-dimensional path-by-sample matrix")
    return AbstractArray(shape=target_alphas.shape, dtype="float64")


def witness_lars_cv_finite_row_mask(mse_path: AbstractArray) -> AbstractArray:
    """Describe a boolean mask over alpha rows of the fold-MSE matrix."""
    if len(mse_path.shape) != 2:
        raise ValueError("mse_path must be two-dimensional")
    return AbstractArray(shape=(int(mse_path.shape[0]),), dtype="bool")


def witness_lars_cv_best_alpha(
    cv_alphas: AbstractArray,
    mse_path: AbstractArray,
) -> float:
    """Describe the scalar alpha chosen from a finite shared MSE path."""
    if len(cv_alphas.shape) != 1 or len(mse_path.shape) != 2:
        raise ValueError("cv_alphas must be one-dimensional and mse_path must be two-dimensional")
    if cv_alphas.shape[0] != mse_path.shape[0]:
        raise ValueError("cv_alphas must align with mse_path rows")
    return 0.0
