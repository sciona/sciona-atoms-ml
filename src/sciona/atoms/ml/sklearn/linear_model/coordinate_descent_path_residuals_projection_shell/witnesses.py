"""Ghost witnesses for sklearn coordinate-descent path-residual projection atoms."""

from __future__ import annotations


def witness_cd_path_residuals_project_test_coefs(X_test: object, coefs: object) -> object:
    """Describe the held-out safe_sparse_dot projection in _path_residuals."""
    return X_test, coefs
