from __future__ import annotations

import pytest
from icontract import ViolationError
from sklearn.linear_model import ElasticNet


EXPECTED_NAMES = (
    "alpha",
    "l1_ratio",
    "fit_intercept",
    "precompute",
    "max_iter",
    "copy_X",
    "tol",
    "warm_start",
    "positive",
    "random_state",
    "selection",
)


def test_coordinate_descent_elastic_net_class_api_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_elastic_net_class_api_shell import (
        cd_elastic_net_fit_metadata_request,
        cd_elastic_net_parameter_constraint_descriptors,
        cd_elastic_net_parameter_constraint_names,
    )

    assert callable(cd_elastic_net_fit_metadata_request)
    assert callable(cd_elastic_net_parameter_constraint_names)
    assert callable(cd_elastic_net_parameter_constraint_descriptors)


def test_coordinate_descent_elastic_net_class_api_shell_matches_sklearn_class_schema() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_elastic_net_class_api_shell import (
        cd_elastic_net_fit_metadata_request,
        cd_elastic_net_parameter_constraint_descriptors,
        cd_elastic_net_parameter_constraint_names,
    )

    assert cd_elastic_net_fit_metadata_request("elastic_net") == {"check_input": "UNUSED"}
    assert cd_elastic_net_parameter_constraint_names("elastic_net") == tuple(
        ElasticNet._parameter_constraints
    )
    assert cd_elastic_net_parameter_constraint_names("elastic_net") == EXPECTED_NAMES

    descriptors = cd_elastic_net_parameter_constraint_descriptors("elastic_net")
    assert tuple(descriptors) == EXPECTED_NAMES
    assert descriptors["alpha"] == (("interval", "Real", 0, None, "left"),)
    assert descriptors["l1_ratio"] == (("interval", "Real", 0, 1, "both"),)
    assert descriptors["precompute"] == ("boolean", "array-like")
    assert descriptors["max_iter"] == (("interval", "Integral", 1, None, "left"), None)
    assert descriptors["selection"] == (("str_options", ("cyclic", "random")),)


def test_coordinate_descent_elastic_net_class_api_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_elastic_net_class_api_shell import (
        cd_elastic_net_fit_metadata_request,
        cd_elastic_net_parameter_constraint_descriptors,
        cd_elastic_net_parameter_constraint_names,
    )

    with pytest.raises(ViolationError):
        cd_elastic_net_fit_metadata_request("lasso")

    with pytest.raises(ViolationError):
        cd_elastic_net_parameter_constraint_names("lasso")

    with pytest.raises(ViolationError):
        cd_elastic_net_parameter_constraint_descriptors("lasso")
