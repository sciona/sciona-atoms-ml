from __future__ import annotations

import pytest
from icontract import ViolationError


def test_coordinate_descent_cv_target_guards_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_target_guards import (
        cd_cv_multitask_monotask_guard_required,
        cd_cv_multitask_monotask_message,
        cd_cv_multitask_sparse_guard_required,
        cd_cv_multitask_sparse_message,
        cd_cv_non_multitask_guard_required,
        cd_cv_non_multitask_message,
        cd_cv_reference_preserving_validation_branch,
        cd_cv_scalar_sample_weight_becomes_none,
    )

    assert callable(cd_cv_reference_preserving_validation_branch)
    assert callable(cd_cv_non_multitask_guard_required)
    assert callable(cd_cv_non_multitask_message)
    assert callable(cd_cv_multitask_sparse_guard_required)
    assert callable(cd_cv_multitask_sparse_message)
    assert callable(cd_cv_multitask_monotask_guard_required)
    assert callable(cd_cv_multitask_monotask_message)
    assert callable(cd_cv_scalar_sample_weight_becomes_none)


def test_coordinate_descent_cv_target_guards_match_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_target_guards import (
        cd_cv_multitask_monotask_guard_required,
        cd_cv_multitask_monotask_message,
        cd_cv_multitask_sparse_guard_required,
        cd_cv_multitask_sparse_message,
        cd_cv_non_multitask_guard_required,
        cd_cv_non_multitask_message,
        cd_cv_reference_preserving_validation_branch,
        cd_cv_scalar_sample_weight_becomes_none,
    )

    assert cd_cv_reference_preserving_validation_branch(True, False) is True
    assert cd_cv_non_multitask_guard_required(False, 2, 3) is True
    assert cd_cv_non_multitask_message("ElasticNetCV") == (
        "For multi-task outputs, use MultiTaskElasticNetCV"
    )
    assert cd_cv_multitask_sparse_guard_required(True, True) is True
    assert cd_cv_multitask_sparse_message(True) == (
        "X should be dense but a sparse matrix waspassed"
    )
    assert cd_cv_multitask_monotask_guard_required(True, 1) is True
    assert cd_cv_multitask_monotask_message("MultiTaskElasticNet") == (
        "For mono-task outputs, use ElasticNetCV"
    )
    assert cd_cv_scalar_sample_weight_becomes_none(True) is True


def test_coordinate_descent_cv_target_guards_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_target_guards import (
        cd_cv_multitask_monotask_message,
        cd_cv_non_multitask_guard_required,
    )

    with pytest.raises(ViolationError):
        cd_cv_multitask_monotask_message("ElasticNetCV")

    with pytest.raises(ViolationError):
        cd_cv_non_multitask_guard_required(False, 0, 1)
