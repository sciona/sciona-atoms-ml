from __future__ import annotations

from collections.abc import Mapping
import inspect

import pytest
from icontract import ViolationError
from sklearn.linear_model._coordinate_descent import LinearModelCV


EXPECTED_METHODS = ("_get_estimator", "_is_multitask", "path")


def test_coordinate_descent_cv_base_abstract_api_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_base_abstract_api_shell import (
        cd_cv_base_abstract_method_names,
        cd_cv_base_abstract_method_roles,
        cd_cv_base_path_signature_payload,
    )

    assert callable(cd_cv_base_abstract_method_names)
    assert callable(cd_cv_base_abstract_method_roles)
    assert callable(cd_cv_base_path_signature_payload)


def test_coordinate_descent_cv_base_abstract_api_shell_matches_sklearn_abstract_methods() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_base_abstract_api_shell import (
        cd_cv_base_abstract_method_names,
        cd_cv_base_abstract_method_roles,
    )

    assert cd_cv_base_abstract_method_names("LinearModelCV") == EXPECTED_METHODS
    assert set(cd_cv_base_abstract_method_names("LinearModelCV")) <= LinearModelCV.__abstractmethods__
    assert cd_cv_base_abstract_method_roles("_get_estimator") == "refit_estimator_factory"
    assert cd_cv_base_abstract_method_roles("_is_multitask") == "target_shape_policy"
    assert cd_cv_base_abstract_method_roles("path") == "coordinate_descent_path_callable"


def test_coordinate_descent_cv_base_abstract_api_shell_path_signature_payload() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_base_abstract_api_shell import (
        cd_cv_base_path_signature_payload,
    )

    signature = inspect.signature(LinearModelCV.path)
    assert tuple(signature.parameters) == ("X", "y", "kwargs")
    assert signature.parameters["kwargs"].kind is inspect.Parameter.VAR_KEYWORD

    X = object()
    y = object()
    kwargs: Mapping[str, object] = {"alpha": 0.5, "copy_X": False}
    payload = cd_cv_base_path_signature_payload(X, y, kwargs)

    assert payload["X"] is X
    assert payload["y"] is y
    assert payload["kwargs"] == kwargs
    assert payload["kwargs"] is not kwargs


def test_coordinate_descent_cv_base_abstract_api_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_base_abstract_api_shell import (
        cd_cv_base_abstract_method_names,
        cd_cv_base_abstract_method_roles,
        cd_cv_base_path_signature_payload,
    )

    with pytest.raises(ViolationError):
        cd_cv_base_abstract_method_names("ElasticNetCV")

    with pytest.raises(ViolationError):
        cd_cv_base_abstract_method_roles("fit")

    with pytest.raises(ViolationError):
        cd_cv_base_path_signature_payload(object(), object(), [("alpha", 0.5)])  # type: ignore[arg-type]
