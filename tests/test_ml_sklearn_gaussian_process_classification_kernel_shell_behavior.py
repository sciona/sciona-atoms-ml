from __future__ import annotations

from sklearn.gaussian_process.kernels import CompoundKernel, RBF


def test_gpc_kernel_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_kernel_shell import (
        gpc_kernel_result,
        gpc_kernel_use_binary_branch,
    )

    assert callable(gpc_kernel_use_binary_branch)
    assert callable(gpc_kernel_result)


def test_gpc_kernel_use_binary_branch_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_kernel_shell import (
        gpc_kernel_use_binary_branch,
    )

    assert gpc_kernel_use_binary_branch(2) is True
    assert gpc_kernel_use_binary_branch(3) is False


def test_gpc_kernel_result_matches_binary_and_multiclass_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_kernel_shell import (
        gpc_kernel_result,
    )

    binary_kernel = RBF(length_scale=1.25)
    assert gpc_kernel_result(2, binary_kernel=binary_kernel, estimator_kernels=(binary_kernel,)) is binary_kernel

    kernels = (RBF(length_scale=1.0), RBF(length_scale=2.0), RBF(length_scale=3.0))
    observed = gpc_kernel_result(3, estimator_kernels=kernels)

    assert isinstance(observed, CompoundKernel)
    assert len(observed.kernels) == 3
    assert all(observed.kernels[i] is kernels[i] for i in range(3))
