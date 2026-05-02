"""Ghost witnesses for Gaussian-process regression fit-shell atoms."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def witness_gp_fit_dtype_name(
    kernel_requires_vector_input: bool,
) -> str | None:
    """Describe the dtype mode passed into sklearn validation for GP fitting."""
    del kernel_requires_vector_input
    return None


def witness_gp_fit_validate_ensure_2d(
    kernel_requires_vector_input: bool,
) -> bool:
    """Describe the ensure_2d mode passed into sklearn validation for GP fitting."""
    del kernel_requires_vector_input
    return False


def witness_gp_fit_use_optimizer_branch(
    optimizer_is_not_none: bool,
    kernel_n_dims: int,
) -> bool:
    """Describe whether GaussianProcessRegressor.fit enters optimizer selection."""
    del optimizer_is_not_none
    del kernel_n_dims
    return False


def witness_gp_fit_stored_train_inputs(
    X: NDArray[np.float64],
    copy_X_train: bool,
) -> NDArray[np.float64]:
    """Describe the training input matrix stored during Gaussian-process fitting."""
    del X
    del copy_X_train
    return np.zeros((1, 1), dtype=np.float64)


def witness_gp_fit_stored_train_targets(
    y: NDArray[np.float64],
    copy_X_train: bool,
) -> NDArray[np.float64]:
    """Describe the training targets stored during Gaussian-process fitting."""
    del y
    del copy_X_train
    return np.zeros(1, dtype=np.float64)
