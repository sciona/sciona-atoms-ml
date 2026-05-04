"""Binary Gaussian-process classification prediction output atoms."""

from .atoms import (
    gpc_binary_predict_labels,
    gpc_binary_predict_positive_class_mask,
)

__all__ = [
    "gpc_binary_predict_positive_class_mask",
    "gpc_binary_predict_labels",
]
