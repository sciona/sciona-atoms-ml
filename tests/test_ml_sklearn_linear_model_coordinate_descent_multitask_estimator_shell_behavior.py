from __future__ import annotations

import pytest
from icontract import ViolationError


def test_coordinate_descent_multitask_estimator_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_multitask_estimator_shell import (
        cd_multitask_dual_gap,
        cd_multitask_fit_return_self,
        cd_multitask_model_name,
        cd_multitask_mono_task_guard_required,
        cd_multitask_mono_task_message,
        cd_multitask_random_selection,
        cd_multitask_sparse_input_tag,
        cd_multitask_target_multi_output_tag,
        cd_multitask_target_single_output_tag,
    )

    assert callable(cd_multitask_model_name)
    assert callable(cd_multitask_mono_task_guard_required)
    assert callable(cd_multitask_mono_task_message)
    assert callable(cd_multitask_random_selection)
    assert callable(cd_multitask_dual_gap)
    assert callable(cd_multitask_fit_return_self)
    assert callable(cd_multitask_sparse_input_tag)
    assert callable(cd_multitask_target_multi_output_tag)
    assert callable(cd_multitask_target_single_output_tag)


def test_coordinate_descent_multitask_estimator_shell_matches_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_multitask_estimator_shell import (
        cd_multitask_dual_gap,
        cd_multitask_fit_return_self,
        cd_multitask_model_name,
        cd_multitask_mono_task_guard_required,
        cd_multitask_mono_task_message,
        cd_multitask_random_selection,
        cd_multitask_sparse_input_tag,
        cd_multitask_target_multi_output_tag,
        cd_multitask_target_single_output_tag,
    )

    assert cd_multitask_model_name(True) == "ElasticNet"
    assert cd_multitask_mono_task_guard_required(1) is True
    assert cd_multitask_mono_task_message("Lasso") == "For mono-task outputs, use Lasso"
    assert cd_multitask_random_selection("random") is True
    assert cd_multitask_dual_gap(0.6, 3) == pytest.approx(0.2)
    token = object()
    assert cd_multitask_fit_return_self(token) is token
    assert cd_multitask_sparse_input_tag() is False
    assert cd_multitask_target_multi_output_tag() is True
    assert cd_multitask_target_single_output_tag() is False


def test_coordinate_descent_multitask_estimator_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_multitask_estimator_shell import (
        cd_multitask_dual_gap,
        cd_multitask_random_selection,
    )

    with pytest.raises(ViolationError):
        cd_multitask_random_selection("bad")

    with pytest.raises(ViolationError):
        cd_multitask_dual_gap(1.0, 0)
