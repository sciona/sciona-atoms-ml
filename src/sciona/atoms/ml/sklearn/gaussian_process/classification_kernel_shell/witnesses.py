"""Ghost witnesses for Gaussian-process classification kernel shell atoms."""

from __future__ import annotations

from sklearn.gaussian_process.kernels import CompoundKernel, Kernel


def witness_gpc_kernel_use_binary_branch(
    n_classes: int,
) -> bool:
    """Describe whether GaussianProcessClassifier.kernel_ uses the binary branch."""
    del n_classes
    return False


def witness_gpc_kernel_result(
    n_classes: int,
    *,
    binary_kernel: Kernel | None = None,
    estimator_kernels: tuple[Kernel, ...] = (),
) -> Kernel:
    """Describe the fitted kernel object returned by GaussianProcessClassifier.kernel_."""
    del n_classes
    if binary_kernel is not None:
        return binary_kernel
    return CompoundKernel(list(estimator_kernels))
