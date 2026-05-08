"""Ghost witnesses for sklearn enet_path validation callback atoms."""

from __future__ import annotations


def witness_cd_enet_path_check_array_X_args(X: object) -> object:
    """Describe positional args for enet_path X validation."""
    return X


def witness_cd_enet_path_check_array_X_kwargs(copy_X: object) -> object:
    """Describe kwargs for enet_path X validation."""
    return copy_X


def witness_cd_enet_path_check_array_y_args(y: object) -> object:
    """Describe positional args for enet_path y validation."""
    return y


def witness_cd_enet_path_check_array_y_kwargs(x_dtype_type: object) -> object:
    """Describe kwargs for enet_path y validation."""
    return x_dtype_type


def witness_cd_enet_path_check_array_Xy_args(Xy: object) -> object:
    """Describe positional args for enet_path Xy validation."""
    return Xy


def witness_cd_enet_path_check_array_Xy_kwargs(x_dtype_type: object) -> object:
    """Describe kwargs for enet_path Xy validation."""
    return x_dtype_type


def witness_cd_enet_path_check_array_gram_args(precompute: object) -> object:
    """Describe positional args for enet_path Gram validation."""
    return precompute


def witness_cd_enet_path_check_array_gram_kwargs(x_dtype_type: object) -> object:
    """Describe kwargs for enet_path Gram validation."""
    return x_dtype_type
