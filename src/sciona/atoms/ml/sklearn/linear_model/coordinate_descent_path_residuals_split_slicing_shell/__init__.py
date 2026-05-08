"""Sklearn coordinate-descent path-residual split slicing atoms."""

from .atoms import (
    cd_path_residuals_X_test_slice,
    cd_path_residuals_X_train_slice,
    cd_path_residuals_y_test_slice,
    cd_path_residuals_y_train_slice,
)

__all__ = [
    "cd_path_residuals_X_train_slice",
    "cd_path_residuals_y_train_slice",
    "cd_path_residuals_X_test_slice",
    "cd_path_residuals_y_test_slice",
]
