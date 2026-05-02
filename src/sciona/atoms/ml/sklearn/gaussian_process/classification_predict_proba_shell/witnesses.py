"""Ghost witnesses for Gaussian-process classification predict_proba shell atoms."""

from __future__ import annotations


def witness_gpc_predict_proba_require_supported_multiclass_mode(
    n_classes: int,
    multi_class: str,
) -> bool:
    """Describe whether GaussianProcessClassifier.predict_proba is supported for the fitted multiclass mode."""
    del n_classes
    del multi_class
    return True


def witness_gpc_predict_proba_dtype_name(
    kernel_is_none_or_requires_vector_input: bool,
) -> str | None:
    """Describe the dtype mode passed into sklearn validation for GPC predict_proba."""
    del kernel_is_none_or_requires_vector_input
    return None


def witness_gpc_predict_proba_validate_ensure_2d(
    kernel_is_none_or_requires_vector_input: bool,
) -> bool:
    """Describe the ensure_2d mode passed into sklearn validation for GPC predict_proba."""
    del kernel_is_none_or_requires_vector_input
    return False
