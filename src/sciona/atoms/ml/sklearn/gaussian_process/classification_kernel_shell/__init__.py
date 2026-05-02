"""Gaussian-process classification kernel shell atoms."""

from .atoms import (
    gpc_kernel_result,
    gpc_kernel_use_binary_branch,
)

__all__ = [
    "gpc_kernel_use_binary_branch",
    "gpc_kernel_result",
]
