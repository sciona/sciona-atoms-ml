"""Ghost witnesses for sklearn multitask CV sample-weight absence atoms."""

from __future__ import annotations


def witness_cd_multitask_cv_fit_signature(class_name: object) -> object:
    """Describe the multitask coordinate-descent CV fit signature."""
    return class_name


def witness_cd_multitask_cv_fit_params_name(params_name: object) -> object:
    """Describe the variadic keyword parameter name in multitask CV fit."""
    return params_name


def witness_cd_multitask_cv_sample_weight_absent(parameter_names: object) -> object:
    """Describe sample_weight absence from multitask CV fit parameters."""
    return parameter_names


def witness_cd_multitask_cv_fit_signature_classes(class_names: object) -> object:
    """Describe the multitask CV classes sharing the signature seam."""
    return class_names
