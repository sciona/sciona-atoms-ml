from __future__ import annotations

import pytest
from icontract import ViolationError


def test_coordinate_descent_cv_routing_guards_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_routing_guards import (
        cd_cv_default_routed_params_required,
        cd_cv_drop_estimator_sample_weight,
        cd_cv_forward_splitter_sample_weight,
        cd_cv_routing_enabled_branch,
        cd_cv_sample_weight_support_guard_required,
        cd_cv_sample_weight_support_message,
    )

    assert callable(cd_cv_routing_enabled_branch)
    assert callable(cd_cv_sample_weight_support_guard_required)
    assert callable(cd_cv_sample_weight_support_message)
    assert callable(cd_cv_forward_splitter_sample_weight)
    assert callable(cd_cv_drop_estimator_sample_weight)
    assert callable(cd_cv_default_routed_params_required)


def test_coordinate_descent_cv_routing_guards_match_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_routing_guards import (
        cd_cv_default_routed_params_required,
        cd_cv_drop_estimator_sample_weight,
        cd_cv_forward_splitter_sample_weight,
        cd_cv_routing_enabled_branch,
        cd_cv_sample_weight_support_guard_required,
        cd_cv_sample_weight_support_message,
    )

    assert cd_cv_routing_enabled_branch(True) is True
    assert cd_cv_sample_weight_support_guard_required(True, False, False) is True
    assert cd_cv_sample_weight_support_message(True) == (
        "The CV splitter and underlying estimator do not support sample weights."
    )
    assert cd_cv_forward_splitter_sample_weight(True) is True
    assert cd_cv_drop_estimator_sample_weight(True, False) is True
    assert cd_cv_default_routed_params_required(False) is True


def test_coordinate_descent_cv_routing_guards_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_routing_guards import (
        cd_cv_drop_estimator_sample_weight,
        cd_cv_routing_enabled_branch,
    )

    with pytest.raises(ViolationError):
        cd_cv_routing_enabled_branch("yes")  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_cv_drop_estimator_sample_weight(True, None)  # type: ignore[arg-type]
