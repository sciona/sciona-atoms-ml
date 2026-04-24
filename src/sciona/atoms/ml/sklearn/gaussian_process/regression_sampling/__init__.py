"""Gaussian-process regression sampling helper atoms."""

from .atoms import (
    gp_sample_y_multi_output,
    gp_sample_y_single_output,
)

__all__ = [
    "gp_sample_y_single_output",
    "gp_sample_y_multi_output",
]
