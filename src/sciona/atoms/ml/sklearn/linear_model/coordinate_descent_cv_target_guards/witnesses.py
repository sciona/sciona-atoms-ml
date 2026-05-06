"""Ghost witnesses for sklearn coordinate-descent CV target-guard atoms."""

from __future__ import annotations


def witness_cd_cv_reference_preserving_validation_branch(
    x_is_ndarray: object, x_is_sparse: object
) -> object:
    """Describe the X-type branch used by LinearModelCV.fit validation."""
    return x_is_ndarray, x_is_sparse


def witness_cd_cv_non_multitask_guard_required(
    multitask: object, y_ndim: object, y_width: object
) -> object:
    """Describe the non-multitask multi-output guard in LinearModelCV.fit."""
    return multitask, y_ndim, y_width


def witness_cd_cv_non_multitask_message(class_name: object) -> object:
    """Describe the non-multitask multi-output error message in LinearModelCV.fit."""
    return class_name


def witness_cd_cv_multitask_sparse_guard_required(
    multitask: object, x_is_sparse: object
) -> object:
    """Describe the multitask sparse-input guard in LinearModelCV.fit."""
    return multitask, x_is_sparse


def witness_cd_cv_multitask_sparse_message(guard_required: object) -> object:
    """Describe the multitask sparse-input error message in LinearModelCV.fit."""
    return guard_required


def witness_cd_cv_multitask_monotask_guard_required(
    multitask: object, y_ndim: object
) -> object:
    """Describe the multitask mono-task guard in LinearModelCV.fit."""
    return multitask, y_ndim


def witness_cd_cv_multitask_monotask_message(class_name: object) -> object:
    """Describe the multitask mono-task error message in LinearModelCV.fit."""
    return class_name


def witness_cd_cv_scalar_sample_weight_becomes_none(sample_weight_is_number: object) -> object:
    """Describe the scalar sample-weight normalization branch in LinearModelCV.fit."""
    return sample_weight_is_number
