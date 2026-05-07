"""Deterministic sklearn coordinate-descent CV subclass API-shell atoms."""

from .atoms import (
    cd_cv_subclass_estimator_name,
    cd_cv_subclass_fit_forwards_sample_weight,
    cd_cv_subclass_is_multitask,
    cd_cv_subclass_path_name,
    cd_cv_subclass_super_fit_args,
    cd_cv_subclass_super_fit_kwargs,
    cd_cv_subclass_target_single_output_tag,
)

__all__ = [
    "cd_cv_subclass_path_name",
    "cd_cv_subclass_estimator_name",
    "cd_cv_subclass_is_multitask",
    "cd_cv_subclass_target_single_output_tag",
    "cd_cv_subclass_fit_forwards_sample_weight",
    "cd_cv_subclass_super_fit_args",
    "cd_cv_subclass_super_fit_kwargs",
]
