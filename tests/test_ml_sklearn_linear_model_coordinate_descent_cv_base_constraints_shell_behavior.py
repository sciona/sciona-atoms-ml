from __future__ import annotations

import pytest
from icontract import ViolationError
from sklearn.linear_model._coordinate_descent import LinearModelCV


EXPECTED_NAMES = (
    "eps",
    "n_alphas",
    "alphas",
    "fit_intercept",
    "precompute",
    "max_iter",
    "tol",
    "copy_X",
    "cv",
    "verbose",
    "n_jobs",
    "positive",
    "random_state",
    "selection",
)


def test_coordinate_descent_cv_base_constraints_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_base_constraints_shell import (
        cd_cv_base_parameter_constraint_descriptors,
        cd_cv_base_parameter_constraint_names,
    )

    assert callable(cd_cv_base_parameter_constraint_names)
    assert callable(cd_cv_base_parameter_constraint_descriptors)


def test_coordinate_descent_cv_base_constraints_shell_matches_sklearn_class_schema() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_base_constraints_shell import (
        cd_cv_base_parameter_constraint_descriptors,
        cd_cv_base_parameter_constraint_names,
    )

    assert cd_cv_base_parameter_constraint_names("linear_model_cv") == tuple(
        LinearModelCV._parameter_constraints
    )
    assert cd_cv_base_parameter_constraint_names("linear_model_cv") == EXPECTED_NAMES

    descriptors = cd_cv_base_parameter_constraint_descriptors("linear_model_cv")
    assert tuple(descriptors) == EXPECTED_NAMES
    assert descriptors["eps"] == (("interval", "Real", 0, None, "neither"),)
    assert descriptors["n_alphas"] == (("interval", "Integral", 1, None, "left"),)
    assert descriptors["alphas"] == ("array-like", None)
    assert descriptors["precompute"] == (("str_options", ("auto",)), "array-like", "boolean")
    assert descriptors["max_iter"] == (("interval", "Integral", 1, None, "left"),)
    assert descriptors["tol"] == (("interval", "Real", 0, None, "left"),)
    assert descriptors["cv"] == ("cv_object",)
    assert descriptors["n_jobs"] == ("Integral", None)
    assert descriptors["selection"] == (("str_options", ("cyclic", "random")),)


def test_coordinate_descent_cv_base_constraints_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_base_constraints_shell import (
        cd_cv_base_parameter_constraint_descriptors,
        cd_cv_base_parameter_constraint_names,
    )

    with pytest.raises(ViolationError):
        cd_cv_base_parameter_constraint_names("elastic_net")

    with pytest.raises(ViolationError):
        cd_cv_base_parameter_constraint_descriptors("elastic_net")
