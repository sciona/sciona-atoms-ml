"""Ghost witnesses for sklearn coordinate-descent CV validation-prelude shell atoms."""

from __future__ import annotations


def witness_cd_cv_fit_params_guard_args(params: object, estimator: object) -> object:
    """Describe _raise_for_params(params, self, 'fit') positional args."""
    return params, estimator


def witness_cd_cv_initial_copy_x(copy_x: object, fit_intercept: object) -> object:
    """Describe initial copy_X = self.copy_X and self.fit_intercept."""
    return copy_x, fit_intercept


def witness_cd_cv_check_y_params(validation_required: object) -> object:
    """Describe check_y_params construction before validate_data."""
    return validation_required


def witness_cd_cv_reference_check_x_params(reference_branch_required: object) -> object:
    """Describe check_X_params for ndarray or sparse inputs."""
    return reference_branch_required


def witness_cd_cv_fortran_check_x_params(
    copy_x: object, reference_branch_required: object
) -> object:
    """Describe check_X_params for non-ndarray, non-sparse inputs."""
    return copy_x, reference_branch_required


def witness_cd_cv_reference_validation_copy_x(
    copy_x: object,
    x_is_sparse: object,
    sparse_data_copied: object,
    dense_array_copied: object,
) -> object:
    """Describe copy_X reset after reference-preserving validation."""
    return copy_x, x_is_sparse, sparse_data_copied, dense_array_copied


def witness_cd_cv_non_reference_copy_x(reference_branch_required: object) -> object:
    """Describe copy_X reset after Fortran-order validation."""
    return reference_branch_required
