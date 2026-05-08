"""Gaussian-process classification kernel shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_gpc_kernel_result,
    witness_gpc_kernel_use_binary_branch,
)

def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)

def _kernel(value: object) -> bool:
    from sklearn.gaussian_process.kernels import CompoundKernel, Kernel
    return isinstance(value, Kernel)

def _kernel_sequence(values: object) -> bool:
    return bool(isinstance(values, (list, tuple)) and len(values) >= 1 and all(_kernel(value) for value in values))

@register_atom(witness_gpc_kernel_use_binary_branch)
@icontract.require(lambda n_classes: _positive_int(n_classes), "n_classes must be a positive integer")
def gpc_kernel_use_binary_branch(
    n_classes: int,
) -> bool:
    """Decide whether GaussianProcessClassifier.kernel_ returns the binary base-estimator kernel directly."""
    return int(n_classes) == 2

@register_atom(witness_gpc_kernel_result)
@icontract.require(lambda n_classes: _positive_int(n_classes), "n_classes must be a positive integer")
@icontract.require(lambda binary_kernel=None: binary_kernel is None or _kernel(binary_kernel), "binary_kernel must be a sklearn Kernel when provided")
@icontract.require(lambda estimator_kernels=(): _kernel_sequence(estimator_kernels), "estimator_kernels must be a nonempty sequence of sklearn Kernels")
@icontract.ensure(lambda result: _kernel(result), "result must be a sklearn Kernel")
def gpc_kernel_result(
    n_classes: int,
    *,
    binary_kernel: Kernel | None = None,
    estimator_kernels: tuple[Kernel, ...] = (),
) -> Kernel:
    from sklearn.gaussian_process.kernels import CompoundKernel, Kernel
    """Resolve GaussianProcessClassifier.kernel_ from the fitted binary or multiclass estimator kernels."""
    if gpc_kernel_use_binary_branch(n_classes):
        if binary_kernel is None:
            raise ValueError("binary_kernel must be provided when n_classes == 2")
        return binary_kernel
    return CompoundKernel(list(estimator_kernels))
