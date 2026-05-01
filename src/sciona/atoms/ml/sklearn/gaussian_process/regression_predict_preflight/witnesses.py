"""Ghost witnesses for Gaussian-process regression predict preflight helper atoms."""

from __future__ import annotations


def witness_gp_predict_require_single_uncertainty_mode(
    return_std: bool,
    return_cov: bool,
) -> None:
    """Describe the predict-time guard that rejects requesting std and covariance together."""
    del return_std
    del return_cov
    return None


def witness_gp_predict_dtype_name(
    kernel_is_none: bool,
    kernel_requires_vector_input: bool,
) -> str | None:
    """Describe the dtype mode passed into sklearn validation for GP prediction."""
    del kernel_is_none
    del kernel_requires_vector_input
    return None


def witness_gp_predict_validate_ensure_2d(
    kernel_is_none: bool,
    kernel_requires_vector_input: bool,
) -> bool:
    """Describe the ensure_2d mode passed into sklearn validation for GP prediction."""
    del kernel_is_none
    del kernel_requires_vector_input
    return False


def witness_gp_predict_use_prior_branch(
    has_x_train: bool,
) -> bool:
    """Describe whether GaussianProcessRegressor.predict uses the prior branch."""
    del has_x_train
    return False
