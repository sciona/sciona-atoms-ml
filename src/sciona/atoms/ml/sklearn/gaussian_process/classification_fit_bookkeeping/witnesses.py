"""Ghost witnesses for Gaussian-process classification fit-bookkeeping atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_gpc_fit_require_not_compound_kernel(
    kernel_is_compound: bool,
) -> bool:
    """Describe the CompoundKernel guard in GaussianProcessClassifier.fit."""
    del kernel_is_compound
    return True


def witness_gpc_fit_dtype_name(
    kernel_is_none_or_requires_vector_input: bool,
) -> str | None:
    """Describe the dtype mode passed into sklearn validation for GPC fitting."""
    del kernel_is_none_or_requires_vector_input
    return None


def witness_gpc_fit_validate_ensure_2d(
    kernel_is_none_or_requires_vector_input: bool,
) -> bool:
    """Describe the ensure_2d mode passed into sklearn validation for GPC fitting."""
    del kernel_is_none_or_requires_vector_input
    return False


def witness_gpc_fit_classes(
    y: AbstractArray,
) -> AbstractArray:
    """Describe the sorted unique class vector discovered during GPC fitting."""
    if len(y.shape) != 1:
        raise ValueError("y must be 1D")
    size = int(y.shape[0])
    if size < 1:
        raise ValueError("y must be nonempty")
    return AbstractArray(shape=(size,), dtype="object")


def witness_gpc_fit_class_count(
    classes: AbstractArray,
) -> int:
    """Describe the fitted class count derived from the class vector."""
    if len(classes.shape) != 1:
        raise ValueError("classes must be 1D")
    size = int(classes.shape[0])
    if size < 1:
        raise ValueError("classes must be nonempty")
    return size


def witness_gpc_fit_require_multiple_classes(
    classes: AbstractArray,
) -> int:
    """Describe the validated class count after GPC's distinct-class guard."""
    if len(classes.shape) != 1:
        raise ValueError("classes must be 1D")
    size = int(classes.shape[0])
    if size < 1:
        raise ValueError("classes must be nonempty")
    return size


def witness_gpc_fit_use_one_vs_rest(
    n_classes: int,
    multi_class: str,
) -> bool:
    """Describe the OneVsRestClassifier wrapper branch in GPC fitting."""
    del n_classes
    del multi_class
    return False


def witness_gpc_fit_use_one_vs_one(
    n_classes: int,
    multi_class: str,
) -> bool:
    """Describe the OneVsOneClassifier wrapper branch in GPC fitting."""
    del n_classes
    del multi_class
    return False
