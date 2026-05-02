"""Ghost witnesses for Gaussian-process classification predict shell atoms."""

from __future__ import annotations


def witness_gpc_predict_dtype_name(
    kernel_is_none_or_requires_vector_input: bool,
) -> str | None:
    """Describe the dtype mode passed into sklearn validation for GPC predict."""
    del kernel_is_none_or_requires_vector_input
    return None


def witness_gpc_predict_validate_ensure_2d(
    kernel_is_none_or_requires_vector_input: bool,
) -> bool:
    """Describe the ensure_2d mode passed into sklearn validation for GPC predict."""
    del kernel_is_none_or_requires_vector_input
    return False
