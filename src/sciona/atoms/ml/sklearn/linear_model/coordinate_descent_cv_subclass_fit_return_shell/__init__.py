"""Deterministic sklearn coordinate-descent CV subclass fit-return atoms."""

from .atoms import (
    cd_cv_subclass_fit_returns_super_result,
    cd_cv_subclass_return_passthrough_required,
)

__all__ = [
    "cd_cv_subclass_return_passthrough_required",
    "cd_cv_subclass_fit_returns_super_result",
]
