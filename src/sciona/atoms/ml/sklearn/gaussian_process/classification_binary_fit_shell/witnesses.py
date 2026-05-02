"""Ghost witnesses for binary Gaussian-process classification fit-shell atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_gpc_binary_fit_kernel(
    kernel: object,
) -> object:
    """Describe the default-or-cloned kernel selected during binary GPC fitting."""
    del kernel
    return object()


def witness_gpc_binary_fit_stored_train_inputs(
    X: AbstractArray,
    copy_X_train: bool,
) -> AbstractArray:
    """Describe the stored training inputs after copy_X_train handling."""
    del copy_X_train
    return X


def witness_gpc_binary_fit_classes(
    y: AbstractArray,
) -> AbstractArray:
    """Describe the sorted unique class vector discovered during binary GPC fitting."""
    if len(y.shape) != 1:
        raise ValueError("y must be 1D")
    size = int(y.shape[0])
    if size < 1:
        raise ValueError("y must be nonempty")
    return AbstractArray(shape=(size,), dtype="object")


def witness_gpc_binary_fit_encoded_targets(
    y: AbstractArray,
) -> AbstractArray:
    """Describe the LabelEncoder-style integer target codes."""
    if len(y.shape) != 1:
        raise ValueError("y must be 1D")
    size = int(y.shape[0])
    if size < 1:
        raise ValueError("y must be nonempty")
    return AbstractArray(shape=(size,), dtype="int64")


def witness_gpc_binary_fit_require_binary_classes(
    classes: AbstractArray,
    class_name: str,
) -> int:
    """Describe the validated class count after the binary-class guard."""
    del class_name
    if len(classes.shape) != 1:
        raise ValueError("classes must be 1D")
    size = int(classes.shape[0])
    if size < 1:
        raise ValueError("classes must be nonempty")
    return size


def witness_gpc_binary_fit_use_optimizer_branch(
    optimizer_is_not_none: bool,
    kernel_n_dims: int,
) -> bool:
    """Describe whether binary GPC fitting enters optimizer selection."""
    del optimizer_is_not_none
    del kernel_n_dims
    return False
