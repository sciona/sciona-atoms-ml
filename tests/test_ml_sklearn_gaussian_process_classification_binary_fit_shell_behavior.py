from __future__ import annotations

import numpy as np
import pytest
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF


def test_gpc_binary_fit_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_binary_fit_shell import (
        gpc_binary_fit_classes,
        gpc_binary_fit_encoded_targets,
        gpc_binary_fit_kernel,
        gpc_binary_fit_require_binary_classes,
        gpc_binary_fit_stored_train_inputs,
        gpc_binary_fit_use_optimizer_branch,
    )

    assert callable(gpc_binary_fit_kernel)
    assert callable(gpc_binary_fit_stored_train_inputs)
    assert callable(gpc_binary_fit_classes)
    assert callable(gpc_binary_fit_encoded_targets)
    assert callable(gpc_binary_fit_require_binary_classes)
    assert callable(gpc_binary_fit_use_optimizer_branch)


def test_gpc_binary_fit_kernel_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_binary_fit_shell import (
        gpc_binary_fit_kernel,
    )

    default_kernel = gpc_binary_fit_kernel(None)
    assert default_kernel.__class__.__name__ == "Product"
    assert default_kernel.k1.constant_value == pytest.approx(1.0)
    assert default_kernel.k1.constant_value_bounds == "fixed"
    assert default_kernel.k2.length_scale == pytest.approx(1.0)
    assert default_kernel.k2.length_scale_bounds == "fixed"

    kernel = C(2.0) * RBF(3.0)
    cloned = gpc_binary_fit_kernel(kernel)
    assert cloned is not kernel
    assert cloned.get_params() == kernel.get_params()


def test_gpc_binary_fit_stored_train_inputs_respects_copy_policy() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_binary_fit_shell import (
        gpc_binary_fit_stored_train_inputs,
    )

    X = np.array([[1.0, 2.0], [3.0, 4.0]])

    shared = gpc_binary_fit_stored_train_inputs(X, False)
    copied = gpc_binary_fit_stored_train_inputs(X, True)

    assert shared is X
    assert copied is not X
    assert np.array_equal(shared, X)
    assert np.array_equal(copied, X)


def test_gpc_binary_fit_classes_and_encoding_match_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_binary_fit_shell import (
        gpc_binary_fit_classes,
        gpc_binary_fit_encoded_targets,
    )

    y = np.array(["dog", "cat", "dog", "bird"], dtype=object)

    assert np.array_equal(
        gpc_binary_fit_classes(y),
        np.array(["bird", "cat", "dog"], dtype=object),
    )
    assert np.array_equal(
        gpc_binary_fit_encoded_targets(y),
        np.array([2, 1, 2, 0], dtype=np.int64),
    )


def test_gpc_binary_fit_require_binary_classes_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_binary_fit_shell import (
        gpc_binary_fit_require_binary_classes,
    )

    assert gpc_binary_fit_require_binary_classes(np.array(["a", "b"], dtype=object), "_BinaryGaussianProcessClassifierLaplace") == 2

    with pytest.raises(
        ValueError,
        match=r"_BinaryGaussianProcessClassifierLaplace supports only binary classification\. y contains classes \['a' 'b' 'c'\]",
    ):
        gpc_binary_fit_require_binary_classes(
            np.array(["a", "b", "c"], dtype=object),
            "_BinaryGaussianProcessClassifierLaplace",
        )

    with pytest.raises(
        ValueError,
        match=r"_BinaryGaussianProcessClassifierLaplace requires 2 classes; got 1 class",
    ):
        gpc_binary_fit_require_binary_classes(
            np.array(["only"], dtype=object),
            "_BinaryGaussianProcessClassifierLaplace",
        )


def test_gpc_binary_fit_use_optimizer_branch_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_binary_fit_shell import (
        gpc_binary_fit_use_optimizer_branch,
    )

    assert gpc_binary_fit_use_optimizer_branch(True, 1) is True
    assert gpc_binary_fit_use_optimizer_branch(True, 0) is False
    assert gpc_binary_fit_use_optimizer_branch(False, 3) is False
