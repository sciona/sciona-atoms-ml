"""Ghost witnesses for sklearn coordinate-descent path-residual error-aggregation atoms."""

from __future__ import annotations


def witness_cd_path_residuals_intercepts(y_offset: object, X_offset: object, coefs: object) -> object:
    """Describe the intercept tensor shell in _path_residuals."""
    return y_offset, X_offset, coefs


def witness_cd_path_residuals_residues(X_test_coefs: object, y_test: object, intercepts: object) -> object:
    """Describe the residual tensor shell in _path_residuals."""
    return X_test_coefs, y_test, intercepts


def witness_cd_path_residuals_use_weighted_mse(sample_weight: object) -> object:
    """Describe the sample-weight MSE branch in _path_residuals."""
    return sample_weight


def witness_cd_path_residuals_mse(residues: object, sw_test: object, use_weighted_mse: object) -> object:
    """Describe the per-alpha MSE reduction shell in _path_residuals."""
    return residues, sw_test, use_weighted_mse


def witness_cd_path_residuals_mean_mse(this_mse: object) -> object:
    """Describe the final mean(axis=0) reduction in _path_residuals."""
    return this_mse
