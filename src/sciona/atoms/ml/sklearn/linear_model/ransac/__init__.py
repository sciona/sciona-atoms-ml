"""Estimator-independent sklearn RANSAC consensus helpers."""

from .atoms import (
    ransac_consensus_is_better,
    ransac_default_residual_threshold,
    ransac_dynamic_max_trials,
    ransac_inlier_mask,
    ransac_loss_residuals,
)

__all__ = [
    "ransac_consensus_is_better",
    "ransac_default_residual_threshold",
    "ransac_dynamic_max_trials",
    "ransac_inlier_mask",
    "ransac_loss_residuals",
]
