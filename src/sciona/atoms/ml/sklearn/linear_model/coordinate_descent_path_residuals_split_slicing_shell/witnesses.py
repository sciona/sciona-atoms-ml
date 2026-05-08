"""Ghost witnesses for sklearn coordinate-descent path-residual split slicing atoms."""

from __future__ import annotations


def witness_cd_path_residuals_X_train_slice(X: object, train: object) -> object:
    """Describe the X[train] slicing shell in _path_residuals."""
    return X, train


def witness_cd_path_residuals_y_train_slice(y: object, train: object) -> object:
    """Describe the y[train] slicing shell in _path_residuals."""
    return y, train


def witness_cd_path_residuals_X_test_slice(X: object, test: object) -> object:
    """Describe the X[test] slicing shell in _path_residuals."""
    return X, test


def witness_cd_path_residuals_y_test_slice(y: object, test: object) -> object:
    """Describe the y[test] slicing shell in _path_residuals."""
    return y, test
