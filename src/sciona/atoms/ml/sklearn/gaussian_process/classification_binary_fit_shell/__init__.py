"""Binary Gaussian-process classification fit-shell atoms adapted from scikit-learn."""

from .atoms import (
    gpc_binary_fit_classes,
    gpc_binary_fit_encoded_targets,
    gpc_binary_fit_kernel,
    gpc_binary_fit_require_binary_classes,
    gpc_binary_fit_stored_train_inputs,
    gpc_binary_fit_use_optimizer_branch,
)

__all__ = [
    "gpc_binary_fit_classes",
    "gpc_binary_fit_encoded_targets",
    "gpc_binary_fit_kernel",
    "gpc_binary_fit_require_binary_classes",
    "gpc_binary_fit_stored_train_inputs",
    "gpc_binary_fit_use_optimizer_branch",
]
