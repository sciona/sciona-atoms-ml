from __future__ import annotations

import pytest
from icontract import ViolationError


def test_coordinate_descent_cv_unweighted_refit_callback_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_unweighted_refit_callback_shell import (
        cd_cv_refit_unweighted_fit_args,
        cd_cv_refit_unweighted_fit_call_required,
        cd_cv_refit_unweighted_fit_kwargs,
        cd_cv_refit_unweighted_fitted_model,
    )

    assert callable(cd_cv_refit_unweighted_fit_call_required)
    assert callable(cd_cv_refit_unweighted_fit_args)
    assert callable(cd_cv_refit_unweighted_fit_kwargs)
    assert callable(cd_cv_refit_unweighted_fitted_model)


def test_coordinate_descent_cv_unweighted_refit_callback_shell_matches_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_unweighted_refit_callback_shell import (
        cd_cv_refit_unweighted_fit_args,
        cd_cv_refit_unweighted_fit_call_required,
        cd_cv_refit_unweighted_fit_kwargs,
        cd_cv_refit_unweighted_fitted_model,
    )

    X = object()
    y = object()
    model = object()

    assert cd_cv_refit_unweighted_fit_call_required(None) is True
    assert cd_cv_refit_unweighted_fit_call_required([1.0, 2.0]) is False

    fit_args = cd_cv_refit_unweighted_fit_args(X, y, None)
    assert fit_args == (X, y)
    assert fit_args[0] is X
    assert fit_args[1] is y
    assert cd_cv_refit_unweighted_fit_kwargs(None) == {}
    assert cd_cv_refit_unweighted_fitted_model(model) is model


def test_coordinate_descent_cv_unweighted_refit_callback_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_unweighted_refit_callback_shell import (
        cd_cv_refit_unweighted_fit_args,
        cd_cv_refit_unweighted_fit_kwargs,
    )

    with pytest.raises(ViolationError):
        cd_cv_refit_unweighted_fit_args(object(), object(), [1.0])

    with pytest.raises(ViolationError):
        cd_cv_refit_unweighted_fit_kwargs([1.0])
