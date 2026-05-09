"""Ghost witnesses for sklearn LinearModelCV fit-context atoms."""

from __future__ import annotations


def witness_cd_cv_base_fit_context_kwargs(method_name: object) -> object:
    """Describe the `_fit_context` kwargs applied to LinearModelCV.fit."""
    return method_name


def witness_cd_cv_base_fit_context_method_name(method_name: object) -> object:
    """Describe the method name decorated by `_fit_context`."""
    return method_name
