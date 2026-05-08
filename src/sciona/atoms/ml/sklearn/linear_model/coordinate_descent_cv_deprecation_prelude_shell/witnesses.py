"""Ghost witnesses for sklearn coordinate-descent CV deprecation-prelude atoms."""

from __future__ import annotations


def witness_cd_cv_n_alphas_deprecation_warning_required(n_alphas: object) -> object:
    """Describe the n_alphas deprecation warning predicate."""
    return n_alphas


def witness_cd_cv_n_alphas_deprecation_message(warning_required: object) -> object:
    """Describe the n_alphas deprecation warning message."""
    return warning_required


def witness_cd_cv_alphas_warn_sentinel(alphas: object) -> object:
    """Describe the alphas='warn' sentinel predicate."""
    return alphas


def witness_cd_cv_alphas_none_deprecation_warning_required(alphas: object) -> object:
    """Describe the alphas=None deprecation warning predicate."""
    return alphas


def witness_cd_cv_alphas_none_deprecation_message(warning_required: object) -> object:
    """Describe the alphas=None deprecation warning message."""
    return warning_required


def witness_cd_cv_resolved_alphas(n_alphas: object, alphas: object) -> object:
    """Describe private _alphas resolution before alpha-grid bookkeeping."""
    return n_alphas, alphas
