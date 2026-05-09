from __future__ import annotations

import inspect

import pytest
from icontract import ViolationError
from sklearn.linear_model import MultiTaskElasticNetCV, MultiTaskLassoCV


EXPECTED_SIGNATURE = ("self", "X", "y", "params")
EXPECTED_CLASSES = ("MultiTaskElasticNetCV", "MultiTaskLassoCV")


def test_coordinate_descent_multitask_cv_sample_weight_absence_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_multitask_cv_sample_weight_absence_shell import (
        cd_multitask_cv_fit_params_name,
        cd_multitask_cv_fit_signature,
        cd_multitask_cv_fit_signature_classes,
        cd_multitask_cv_sample_weight_absent,
    )

    assert callable(cd_multitask_cv_fit_signature)
    assert callable(cd_multitask_cv_fit_params_name)
    assert callable(cd_multitask_cv_sample_weight_absent)
    assert callable(cd_multitask_cv_fit_signature_classes)


def test_coordinate_descent_multitask_cv_sample_weight_absence_shell_matches_sklearn_signatures() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_multitask_cv_sample_weight_absence_shell import (
        cd_multitask_cv_fit_params_name,
        cd_multitask_cv_fit_signature,
        cd_multitask_cv_fit_signature_classes,
        cd_multitask_cv_sample_weight_absent,
    )

    for estimator_cls in (MultiTaskElasticNetCV, MultiTaskLassoCV):
        parameter_names = tuple(inspect.signature(estimator_cls.fit).parameters)
        assert parameter_names == EXPECTED_SIGNATURE
        assert cd_multitask_cv_fit_signature(estimator_cls.__name__) == parameter_names
        assert cd_multitask_cv_fit_params_name(parameter_names[-1]) is True
        assert cd_multitask_cv_sample_weight_absent(parameter_names) is True

    assert (
        cd_multitask_cv_fit_signature_classes(
            ["LinearModelCV", "MultiTaskElasticNetCV", "MultiTaskLassoCV"]
        )
        == EXPECTED_CLASSES
    )


def test_coordinate_descent_multitask_cv_sample_weight_absence_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_multitask_cv_sample_weight_absence_shell import (
        cd_multitask_cv_fit_params_name,
        cd_multitask_cv_fit_signature,
        cd_multitask_cv_fit_signature_classes,
        cd_multitask_cv_sample_weight_absent,
    )

    with pytest.raises(ViolationError):
        cd_multitask_cv_fit_signature("ElasticNetCV")

    assert cd_multitask_cv_fit_params_name("sample_weight") is False
    assert cd_multitask_cv_sample_weight_absent(("self", "X", "y", "sample_weight")) is False

    with pytest.raises(ViolationError):
        cd_multitask_cv_sample_weight_absent("self, X, y, params")

    with pytest.raises(ValueError):
        cd_multitask_cv_fit_signature_classes(["MultiTaskElasticNetCV"])
