from __future__ import annotations

import pytest
from icontract import ViolationError


def test_coordinate_descent_cv_subclass_api_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_subclass_api_shell import (
        cd_cv_subclass_estimator_name,
        cd_cv_subclass_fit_forwards_sample_weight,
        cd_cv_subclass_is_multitask,
        cd_cv_subclass_path_name,
        cd_cv_subclass_super_fit_args,
        cd_cv_subclass_super_fit_kwargs,
        cd_cv_subclass_target_single_output_tag,
    )

    assert callable(cd_cv_subclass_path_name)
    assert callable(cd_cv_subclass_estimator_name)
    assert callable(cd_cv_subclass_is_multitask)
    assert callable(cd_cv_subclass_target_single_output_tag)
    assert callable(cd_cv_subclass_fit_forwards_sample_weight)
    assert callable(cd_cv_subclass_super_fit_args)
    assert callable(cd_cv_subclass_super_fit_kwargs)


def test_coordinate_descent_cv_subclass_api_shell_matches_sklearn_subclass_constants() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_subclass_api_shell import (
        cd_cv_subclass_estimator_name,
        cd_cv_subclass_fit_forwards_sample_weight,
        cd_cv_subclass_is_multitask,
        cd_cv_subclass_path_name,
        cd_cv_subclass_target_single_output_tag,
    )

    assert cd_cv_subclass_path_name("lasso_cv") == "lasso_path"
    assert cd_cv_subclass_estimator_name("lasso_cv") == "Lasso"
    assert cd_cv_subclass_is_multitask("lasso_cv") is False
    assert cd_cv_subclass_fit_forwards_sample_weight(False) is True

    assert cd_cv_subclass_path_name("elastic_net_cv") == "enet_path"
    assert cd_cv_subclass_estimator_name("elastic_net_cv") == "ElasticNet"
    assert cd_cv_subclass_is_multitask("elastic_net_cv") is False

    assert cd_cv_subclass_path_name("multi_task_lasso_cv") == "lasso_path"
    assert cd_cv_subclass_estimator_name("multi_task_lasso_cv") == "MultiTaskLasso"
    assert cd_cv_subclass_is_multitask("multi_task_lasso_cv") is True
    assert cd_cv_subclass_target_single_output_tag(True) is False
    assert cd_cv_subclass_fit_forwards_sample_weight(True) is False

    assert cd_cv_subclass_path_name("multi_task_elastic_net_cv") == "enet_path"
    assert cd_cv_subclass_estimator_name("multi_task_elastic_net_cv") == "MultiTaskElasticNet"
    assert cd_cv_subclass_is_multitask("multi_task_elastic_net_cv") is True


def test_coordinate_descent_cv_subclass_api_shell_super_fit_payloads() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_subclass_api_shell import (
        cd_cv_subclass_super_fit_args,
        cd_cv_subclass_super_fit_kwargs,
    )

    X = object()
    y = object()
    args = cd_cv_subclass_super_fit_args(X, y)
    assert args == (X, y)
    assert args[0] is X
    assert args[1] is y

    marker = object()
    params = {"groups": marker}
    weighted = cd_cv_subclass_super_fit_kwargs(params, None, True)
    assert weighted["groups"] is marker
    assert "sample_weight" in weighted
    assert weighted["sample_weight"] is None

    sample_weight = object()
    unweighted = cd_cv_subclass_super_fit_kwargs(params, sample_weight, False)
    assert unweighted == params
    assert "sample_weight" not in unweighted
    assert unweighted["groups"] is marker


def test_coordinate_descent_cv_subclass_api_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_subclass_api_shell import (
        cd_cv_subclass_fit_forwards_sample_weight,
        cd_cv_subclass_path_name,
        cd_cv_subclass_super_fit_kwargs,
        cd_cv_subclass_target_single_output_tag,
    )

    with pytest.raises(ViolationError):
        cd_cv_subclass_path_name("ridge_cv")

    with pytest.raises(ViolationError):
        cd_cv_subclass_target_single_output_tag(False)

    with pytest.raises(ViolationError):
        cd_cv_subclass_fit_forwards_sample_weight("yes")  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_cv_subclass_super_fit_kwargs([], None, True)  # type: ignore[arg-type]
