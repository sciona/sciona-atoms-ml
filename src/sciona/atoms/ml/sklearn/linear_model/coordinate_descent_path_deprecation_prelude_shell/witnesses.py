"""Ghost witnesses for sklearn coordinate-descent path deprecation-prelude atoms."""

from __future__ import annotations


def witness_cd_path_default_n_alphas_resolution(n_alphas: object) -> object:
    """Describe the path-helper n_alphas default branch."""
    return n_alphas


def witness_cd_path_n_alphas_warning_required(n_alphas: object) -> object:
    """Describe the explicit n_alphas deprecation warning predicate."""
    return n_alphas


def witness_cd_path_n_alphas_warning_message(function_name: object) -> object:
    """Describe the exact explicit n_alphas FutureWarning message."""
    return function_name


def witness_cd_path_alphas_none_warning_required(alphas: object) -> object:
    """Describe the alphas=None deprecation warning predicate."""
    return alphas


def witness_cd_path_alphas_none_warning_message(function_name: object) -> object:
    """Describe the exact alphas=None FutureWarning message."""
    return function_name


def witness_cd_path_effective_alphas_resolution(n_alphas: object, alphas: object) -> object:
    """Describe the effective _alphas branch shared by path helpers."""
    return n_alphas, alphas
