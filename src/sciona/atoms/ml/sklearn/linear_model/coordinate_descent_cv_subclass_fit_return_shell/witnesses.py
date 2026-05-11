"""Ghost witnesses for sklearn coordinate-descent CV subclass fit-return atoms."""

from __future__ import annotations


def witness_cd_cv_subclass_return_passthrough_required(cv_kind: object) -> object:
    """Describe whether a CV subclass fit wrapper returns the super-fit result."""
    return cv_kind


def witness_cd_cv_subclass_fit_returns_super_result(
    cv_kind: object,
    super_fit_result: object,
) -> object:
    """Describe identity-preserving return of a delegated super-fit result."""
    return cv_kind, super_fit_result
