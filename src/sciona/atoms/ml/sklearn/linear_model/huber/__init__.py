"""Optimizer-independent sklearn Huber-regression helpers."""

from .atoms import (
    huber_linear_residuals,
    huber_loss_gradient,
    huber_outlier_mask,
)

__all__ = [
    "huber_linear_residuals",
    "huber_loss_gradient",
    "huber_outlier_mask",
]
