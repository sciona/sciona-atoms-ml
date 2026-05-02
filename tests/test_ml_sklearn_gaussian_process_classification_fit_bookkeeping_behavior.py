from __future__ import annotations

import numpy as np
import pytest


def test_gpc_fit_bookkeeping_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_fit_bookkeeping import (
        gpc_fit_class_count,
        gpc_fit_classes,
        gpc_fit_dtype_name,
        gpc_fit_require_multiple_classes,
        gpc_fit_require_not_compound_kernel,
        gpc_fit_use_one_vs_one,
        gpc_fit_use_one_vs_rest,
        gpc_fit_validate_ensure_2d,
    )

    assert callable(gpc_fit_require_not_compound_kernel)
    assert callable(gpc_fit_dtype_name)
    assert callable(gpc_fit_validate_ensure_2d)
    assert callable(gpc_fit_classes)
    assert callable(gpc_fit_class_count)
    assert callable(gpc_fit_require_multiple_classes)
    assert callable(gpc_fit_use_one_vs_rest)
    assert callable(gpc_fit_use_one_vs_one)


def test_gpc_fit_compound_kernel_guard_and_validation_mode_match_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_fit_bookkeeping import (
        gpc_fit_dtype_name,
        gpc_fit_require_not_compound_kernel,
        gpc_fit_validate_ensure_2d,
    )

    assert gpc_fit_require_not_compound_kernel(False) is True
    with pytest.raises(ValueError, match="kernel cannot be a CompoundKernel"):
        gpc_fit_require_not_compound_kernel(True)

    assert gpc_fit_dtype_name(True) == "numeric"
    assert gpc_fit_validate_ensure_2d(True) is True
    assert gpc_fit_dtype_name(False) is None
    assert gpc_fit_validate_ensure_2d(False) is False


def test_gpc_fit_class_bookkeeping_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_fit_bookkeeping import (
        gpc_fit_class_count,
        gpc_fit_classes,
        gpc_fit_require_multiple_classes,
    )

    y = np.array([2, 1, 2, 0, 1], dtype=np.int64)
    classes = gpc_fit_classes(y)

    assert np.array_equal(classes, np.array([0, 1, 2], dtype=np.int64))
    assert gpc_fit_class_count(classes) == 3
    assert gpc_fit_require_multiple_classes(classes) == 3


def test_gpc_fit_single_class_guard_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_fit_bookkeeping import (
        gpc_fit_require_multiple_classes,
    )

    with pytest.raises(
        ValueError,
        match=r"GaussianProcessClassifier requires 2 or more distinct classes; got 1 class \(only class 7 is present\)",
    ):
        gpc_fit_require_multiple_classes(np.array([7], dtype=np.int64))


def test_gpc_fit_multiclass_wrapper_branches_match_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_fit_bookkeeping import (
        gpc_fit_use_one_vs_one,
        gpc_fit_use_one_vs_rest,
    )

    assert gpc_fit_use_one_vs_rest(3, "one_vs_rest") is True
    assert gpc_fit_use_one_vs_rest(2, "one_vs_rest") is False
    assert gpc_fit_use_one_vs_rest(4, "one_vs_one") is False

    assert gpc_fit_use_one_vs_one(3, "one_vs_one") is True
    assert gpc_fit_use_one_vs_one(2, "one_vs_one") is False
    assert gpc_fit_use_one_vs_one(4, "one_vs_rest") is False
