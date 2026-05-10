"""Ghost witnesses for coordinate-descent path validation-decorator atoms."""

from __future__ import annotations


def witness_cd_lasso_path_validation_param_names(path_name: object) -> object:
    """Describe the ordered parameter keys in lasso_path validate_params."""
    return path_name


def witness_cd_enet_path_validation_param_names(path_name: object) -> object:
    """Describe the ordered parameter keys in enet_path validate_params."""
    return path_name


def witness_cd_path_validation_param_descriptors(path_name: object) -> object:
    """Describe compact validate_params descriptors for a path helper."""
    return path_name


def witness_cd_path_validation_prefers_skip_nested(path_name: object) -> object:
    """Describe the shared prefer_skip_nested_validation decorator flag."""
    return path_name
