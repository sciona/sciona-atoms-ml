"""Standard Scaler family containing compute_mean_and_variance and apply_z_scaling."""

from .atoms import (
    compute_mean_and_variance,
    apply_z_scaling,
)

__all__ = [
    "compute_mean_and_variance",
    "apply_z_scaling",
]
