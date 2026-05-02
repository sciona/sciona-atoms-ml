"""Gaussian-process classification fit-bookkeeping atoms."""

from .atoms import (
    gpc_fit_class_count,
    gpc_fit_classes,
    gpc_fit_dtype_name,
    gpc_fit_require_multiple_classes,
    gpc_fit_require_not_compound_kernel,
    gpc_fit_use_one_vs_one,
    gpc_fit_use_one_vs_rest,
    gpc_fit_validate_ensure_2d,
)

__all__ = [
    "gpc_fit_require_not_compound_kernel",
    "gpc_fit_dtype_name",
    "gpc_fit_validate_ensure_2d",
    "gpc_fit_classes",
    "gpc_fit_class_count",
    "gpc_fit_require_multiple_classes",
    "gpc_fit_use_one_vs_rest",
    "gpc_fit_use_one_vs_one",
]
