from __future__ import annotations

import pytest
from icontract import ViolationError


def test_coordinate_descent_elastic_net_dense_decision_callback_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_elastic_net_dense_decision_callback_shell import (
        cd_elastic_net_check_is_fitted_args,
        cd_elastic_net_dense_decision_required,
        cd_elastic_net_dense_super_decision_args,
        cd_elastic_net_dense_super_decision_result,
    )

    assert callable(cd_elastic_net_check_is_fitted_args)
    assert callable(cd_elastic_net_dense_decision_required)
    assert callable(cd_elastic_net_dense_super_decision_args)
    assert callable(cd_elastic_net_dense_super_decision_result)


def test_coordinate_descent_elastic_net_dense_decision_callback_shell_matches_dense_branch() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_elastic_net_dense_decision_callback_shell import (
        cd_elastic_net_check_is_fitted_args,
        cd_elastic_net_dense_decision_required,
        cd_elastic_net_dense_super_decision_args,
        cd_elastic_net_dense_super_decision_result,
    )

    estimator = object()
    X = object()
    decision_result = object()

    assert cd_elastic_net_check_is_fitted_args(estimator) == (estimator,)
    assert cd_elastic_net_dense_decision_required(False) is True
    assert cd_elastic_net_dense_decision_required(True) is False
    assert cd_elastic_net_dense_super_decision_args(estimator, X) == (estimator, X)
    assert cd_elastic_net_dense_super_decision_result(decision_result) is decision_result


def test_coordinate_descent_elastic_net_dense_decision_callback_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_elastic_net_dense_decision_callback_shell import (
        cd_elastic_net_check_is_fitted_args,
        cd_elastic_net_dense_decision_required,
        cd_elastic_net_dense_super_decision_args,
        cd_elastic_net_dense_super_decision_result,
    )

    with pytest.raises(ViolationError):
        cd_elastic_net_check_is_fitted_args(None)

    with pytest.raises(ViolationError):
        cd_elastic_net_dense_decision_required("no")  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_elastic_net_dense_super_decision_args(object(), None)

    with pytest.raises(ViolationError):
        cd_elastic_net_dense_super_decision_result(None)
