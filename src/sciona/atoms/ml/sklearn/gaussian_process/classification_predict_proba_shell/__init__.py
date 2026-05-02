"""Gaussian-process classification predict_proba shell atoms."""

from .atoms import (
    gpc_predict_proba_dtype_name,
    gpc_predict_proba_require_supported_multiclass_mode,
    gpc_predict_proba_validate_ensure_2d,
)

__all__ = [
    "gpc_predict_proba_require_supported_multiclass_mode",
    "gpc_predict_proba_dtype_name",
    "gpc_predict_proba_validate_ensure_2d",
]
